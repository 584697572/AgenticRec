"""Input representation compatibility for the unchanged upstream movie_map function."""


def adapt_release_dates(catalog):
    import pandas as pd
    result = catalog.copy(deep=True)
    if pd.api.types.is_integer_dtype(result["release_date"].dtype):
        # The resource only knows a year. Jan 1 is a comparison representation,
        # not a newly asserted release date; original source data is unchanged.
        result["release_date"] = pd.to_datetime(result["release_date"].astype(str), format="%Y", errors="raise")
    elif not pd.api.types.is_datetime64_any_dtype(result["release_date"].dtype):
        raise ValueError("release_date must contain integer years or datetimes")
    if result["release_date"].isna().any():
        raise ValueError("missing release dates must not be fabricated")
    return result
