import pyspark.sql.functions as F
from pyspark.sql.session import SparkSession
from typing import Tuple, Dict

def get_seasons(spark: SparkSession, query_date: str) -> Tuple[str, Dict[str, Tuple[str, str]]]:
    """
    Retrieves up to 21 season records (including the current season) and returns:
      - The current season identifier (first season based on descending order)
      - A dictionary mapping season names to a tuple of (start_date, end_date) as strings.
    
    The season names are prefixed with 'season' (e.g., "season1", "season2", etc.).
    """
    seasons_df = (
        spark.read.table("main.pubg_meta.pubg_patch_update_date")
        .filter(F.col("device") == "PC")
        .withColumn(
            "patch_name",
            F.split("patch_name", "\.").getItem(0)
        )
        .groupBy("patch_name")
        .agg(
            F.min("patch_target_date").alias("start_date"),
            F.max("patch_end_date").alias("end_date")
        )
        .filter(F.col("start_date") <= query_date)
        .select(
            F.concat(
                F.lit("season"),
                F.col("patch_name")
            ).alias("patch_name"),
            F.col("start_date").cast("string"),
            F.col("end_date").cast("string")
        )
        .orderBy("patch_name", ascending=False)
        .limit(21).orderBy("patch_name")
    )
    seasons_df = seasons_df.toPandas()
    seasons = {
        row.patch_name: (row.start_date, row.end_date) for row in seasons_df.itertuples(index=False)
    }
    q_season = next(reversed(seasons)).split('season')[-1]
    return q_season, seasons