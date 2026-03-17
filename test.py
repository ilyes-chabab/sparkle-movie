"""
PySpark Installation Test Suite
Run with:  python test_pyspark.py
    or:    pytest test_pyspark.py -v
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType
)


# ── Spark session (local mode) ──────────────────────────────
def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("pyspark-test")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


# ── 1. Basic DataFrame creation & operations ────────────────
def test_basic_dataframe():
    spark = get_spark()

    data = [
        ("Alice", "Engineering", 85000),
        ("Bob", "Engineering", 92000),
        ("Carol", "Marketing", 78000),
        ("Dave", "Marketing", 81000),
        ("Eve", "Sales", 74000),
        ("Frank", "Sales", 69000),
    ]

    schema = StructType([
        StructField("name", StringType(), False),
        StructField("department", StringType(), False),
        StructField("salary", IntegerType(), False),
    ])

    df = spark.createDataFrame(data, schema=schema)

    print("\n=== Raw DataFrame ===")
    df.show()
    df.printSchema()

    assert df.count() == 6, "Row count mismatch"
    assert len(df.columns) == 3, "Column count mismatch"
    print("[PASS] Basic DataFrame creation")


# ── 2. Aggregations ─────────────────────────────────────────
def test_aggregations():
    spark = get_spark()

    data = [
        ("Alice", "Engineering", 85000),
        ("Bob", "Engineering", 92000),
        ("Carol", "Marketing", 78000),
        ("Dave", "Marketing", 81000),
        ("Eve", "Sales", 74000),
        ("Frank", "Sales", 69000),
    ]
    df = spark.createDataFrame(data, ["name", "department", "salary"])

    agg_df = (
        df.groupBy("department")
        .agg(
            F.count("name").alias("headcount"),
            F.avg("salary").alias("avg_salary"),
            F.max("salary").alias("max_salary"),
        )
        .orderBy("department")
    )

    print("\n=== Aggregation by Department ===")
    agg_df.show()

    rows = agg_df.collect()
    eng = next(r for r in rows if r["department"] == "Engineering")
    assert eng["headcount"] == 2
    assert eng["avg_salary"] == 88500.0
    print("[PASS] Aggregations")


# ── 3. Spark SQL ─────────────────────────────────────────────
def test_spark_sql():
    spark = get_spark()

    data = [
        ("Alice", "Engineering", 85000),
        ("Bob", "Engineering", 92000),
        ("Carol", "Marketing", 78000),
    ]
    df = spark.createDataFrame(data, ["name", "department", "salary"])
    df.createOrReplaceTempView("employees")

    result = spark.sql("""
        SELECT department, SUM(salary) AS total_salary
        FROM employees
        GROUP BY department
        ORDER BY total_salary DESC
    """)

    print("\n=== Spark SQL Result ===")
    result.show()

    assert result.count() >= 1
    print("[PASS] Spark SQL")


# ── 4. UDF (User Defined Function) ──────────────────────────
def test_udf():
    spark = get_spark()

    data = [("Alice", 85000), ("Bob", 92000), ("Carol", 78000)]
    df = spark.createDataFrame(data, ["name", "salary"])

    @F.udf(StringType())
    def salary_band(salary: int) -> str:
        if salary >= 90000:
            return "Senior"
        elif salary >= 80000:
            return "Mid"
        return "Junior"

    result = df.withColumn("band", salary_band(F.col("salary")))

    print("\n=== UDF Result (salary bands) ===")
    result.show()

    bands = [r["band"] for r in result.collect()]
    assert "Senior" in bands
    assert "Junior" in bands
    print("[PASS] UDF")


# ── 5. Pandas ↔ PySpark interop (requires pyarrow) ──────────
def test_pandas_interop():
    spark = get_spark()

    import pandas as pd

    pdf = pd.DataFrame({
        "product": ["Widget", "Gadget", "Doohickey"],
        "price": [9.99, 24.50, 4.75],
        "quantity": [100, 50, 300],
    })

    # Pandas → Spark
    sdf = spark.createDataFrame(pdf)
    sdf = sdf.withColumn("revenue", F.col("price") * F.col("quantity"))

    print("\n=== Pandas → Spark → computed column ===")
    sdf.show()

    # Spark → Pandas
    back_to_pandas = sdf.toPandas()
    assert "revenue" in back_to_pandas.columns
    assert back_to_pandas.loc[0, "revenue"] == 999.0
    print("[PASS] Pandas interop")


# ── 6. Read / write Parquet ──────────────────────────────────
def test_parquet_io(tmp_path=None):
    import tempfile, os

    spark = get_spark()
    out_dir = tmp_path or tempfile.mkdtemp()
    parquet_path = os.path.join(out_dir, "test_output.parquet")

    data = [(i, f"item_{i}", i * 1.5) for i in range(100)]
    df = spark.createDataFrame(data, ["id", "label", "value"])

    df.write.mode("overwrite").parquet(parquet_path)
    loaded = spark.read.parquet(parquet_path)

    print("\n=== Parquet round-trip ===")
    loaded.show(5)

    assert loaded.count() == 100
    print(f"[PASS] Parquet I/O  ({parquet_path})")


# ── Run all tests ────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  PySpark Installation Test Suite")
    print("=" * 55)

    tests = [
        test_basic_dataframe,
        test_aggregations,
        test_spark_sql,
        test_udf,
        test_pandas_interop,
        test_parquet_io,
    ]

    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"\n[FAIL] {t.__name__}: {e}")
            failed += 1

    # Clean up Spark
    get_spark().stop()

    print("\n" + "=" * 55)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 55)