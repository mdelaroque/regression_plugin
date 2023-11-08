import polars as pl
from functools import partial
from expression_lib import Language, Distance, Regr
from datetime import date
import numpy as np


def regress_multi(s: pl.Series, yvar: str, xvars: list[str]) -> pl.Series:
    df = s.struct.unnest()
    yvar = df[[yvar]].to_numpy()
    xvars = np.hstack([df[xvars].to_numpy(), np.ones((len(s), 1))])
    lsqres = np.linalg.lstsq(xvars, yvar)
    results = lsqres[0][:, 0]
    return pl.Series(results)

df = pl.DataFrame(
    {
        "names": (_n:=["Richard", "Alice", "Bob"]*4),
        "floats": np.random.randn(len(_n)),
        "floats2": np.random.randn(len(_n)),
    }
)

print(df)
try:
    out = df.with_columns((pl.col("floats2")*2).alias("time2")).select(
        pl.col("floats").multireg.ols("floats2", "time2"),
    )
    print(out)
except pl.ComputeError as e:
    print("Groupby failed")
    assert "the plugin failed with message" in str(e)

# 1 Variable

out = df.groupby("names").agg(
    beta=pl.col("floats").multireg.ols(
        "floats2",
        ),
)
print(df.sort(by="names"),"\n",out)

# 2 Variable

out = df.lazy().with_columns(pl.lit(1.).alias("intercept")).groupby("names").agg(
    beta=pl.col("floats").multireg.ols(
        "floats2",
        pl.col("floats")**2,
        pl.col("intercept"),  ## Somehow adding one here kill the core
        # pl.lit(1.).alias("intercept"),  ## Somehow adding one here kill the core, looking at what is passed through to rust, it's only 1 float
        ),
).collect()
print(df.sort(by="names"),"\n",out)

out = df.lazy().with_columns(pl.lit(1.).alias("intercept")).groupby("names").agg(
    beta=pl.col("floats").multireg.ols_ndarray(
        "floats2",
        pl.col("floats")**2,
        pl.col("intercept"),  ## Somehow adding one here kill the core
        # pl.lit(1.).alias("intercept"),  ## Somehow adding one here kill the core, looking at what is passed through to rust, it's only 1 float
        ),
).collect()
print("\n",out)

out = df.lazy().with_columns(pl.lit(1.).alias("intercept")).groupby("names").agg(
    beta=pl.col("floats").multireg.ols_solvenalgebra(
        "floats2",
        pl.col("floats")**2,
        pl.col("intercept"),  ## Somehow adding one here kill the core
        # pl.lit(1.).alias("intercept"),  ## Somehow adding one here kill the core, looking at what is passed through to rust, it's only 1 float
        ),
).collect()
print("\n",out)

out = df.lazy().with_columns(pl.lit(1.).alias("intercept")).groupby("names").agg(
                pl.struct(["floats", "floats2",(pl.col("floats")**2).alias("to_the_power2")]).apply(
                    partial(regress_multi, xvars=["floats2", "to_the_power2"], yvar="floats")
                )
            ).collect()
print("\n",out)
