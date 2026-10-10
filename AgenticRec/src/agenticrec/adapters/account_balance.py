"""Provider-balance policy explicitly authorized for this finite benchmark."""
from decimal import Decimal, InvalidOperation
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from .openai_compatible import HttpResponse, _NoRedirect, _validate_key
from ..runtime.errors import LLMTransportError


class BalanceStop(RuntimeError):
    pass


class BalanceClient:
    def __init__(self, api_key, *, sender=None):
        self._key = _validate_key(api_key)
        self._sender = sender or self._get
        self.query_count = 0

    def __repr__(self):
        return 'BalanceClient(provider=deepseek)'

    def fetch(self):
        self.query_count += 1
        reply = self._sender('https://api.deepseek.com/user/balance',
                             {'Authorization': 'Bearer ' + self._key, 'Accept': 'application/json'}, 15)
        if reply.status_code != 200:
            raise LLMTransportError('balance query failed', status_code=reply.status_code,
                                    error_type='balance_query', retryable=False)
        value = json.loads(reply.body)
        if type(value.get('is_available')) is not bool or type(value.get('balance_infos')) is not list:
            raise ValueError('invalid balance response')
        infos = []
        for row in value['balance_infos']:
            if row.get('currency') not in ('CNY', 'USD'):
                raise ValueError('invalid balance currency')
            safe = {'currency': row['currency']}
            for name in ('total_balance', 'granted_balance', 'topped_up_balance'):
                try:
                    amount = Decimal(row[name])
                except (KeyError, TypeError, InvalidOperation):
                    raise ValueError('invalid balance amount') from None
                if not amount.is_finite() or amount < 0:
                    raise ValueError('invalid balance amount')
                safe[name] = str(amount)
            infos.append(safe)
        return {'is_available': value['is_available'], 'balance_infos': infos}

    @staticmethod
    def _get(url, headers, timeout):
        request = Request(url, headers=headers, method='GET')
        try:
            with build_opener(_NoRedirect()).open(request, timeout=timeout) as stream:
                body = stream.read(65537)
                status = stream.status
        except HTTPError as error:
            body, status = error.read(65537), error.code
        except (URLError, OSError):
            raise LLMTransportError('balance connection failed', error_type='balance_connection',
                                    retryable=False) from None
        if len(body) > 65536:
            raise ValueError('balance response exceeds limit')
        return HttpResponse(status, body)


class BalanceGuard:
    """Latch insufficient balance/auth failures; never top up or retry them."""
    def __init__(self, client, *, refresh_every=25, observer=None):
        self.client = client
        self.refresh_every = refresh_every
        self.observer = observer
        self.requests = 0
        self.halted = False
        self.reason = None
        self.last_balance = None
        self.error_streak = 0
        self.last_checked_requests = 0

    def refresh(self):
        self.last_balance = self.client.fetch()
        self.last_checked_requests = self.requests
        if self.observer:
            self.observer({'event': 'BALANCE', 'after_generation_attempts': self.requests,
                           'balance': self.last_balance, 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        if not self.last_balance['is_available']:
            self.halted, self.reason = True, 'ACCOUNT_BALANCE_INSUFFICIENT'
        return self.last_balance

    def should_stop(self):
        if not self.halted and (self.last_balance is None
                              or self.requests-self.last_checked_requests >= self.refresh_every):
            self.refresh()
        return self.halted

    def call(self, function):
        if self.halted:
            raise BalanceStop(self.reason)
        if self.last_balance is None:
            self.refresh()
        if self.halted:
            raise BalanceStop(self.reason)
        self.requests += 1
        try:
            reply = function()
        except LLMTransportError as error:
            self.error_streak += 1
            if error.status_code in (400, 401, 402, 403, 422):
                self.halted = True
                self.reason = ('ACCOUNT_BALANCE_INSUFFICIENT' if error.status_code == 402
                               else 'PROVIDER_CONFIGURATION_ERROR')
            elif self.error_streak >= 3:
                self.halted, self.reason = True, 'REPEATED_PROVIDER_FAILURE'
            raise
        self.error_streak = 0
        return reply


class GuardedTransport:
    is_live = True
    def __init__(self, transport, guard):
        self.transport, self.guard = transport, guard
    def chat(self, messages, **options):
        return self.guard.call(lambda: self.transport.chat(messages, **options))


if __name__ == '__main__':
    from ..runtime.secrets import read_api_key
    print(json.dumps(BalanceClient(read_api_key()).fetch(), ensure_ascii=True))
