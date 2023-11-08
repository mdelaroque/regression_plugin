import polars as pl
from expression_lib import Language, Distance, Regr
from datetime import date
import numpy as np

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
