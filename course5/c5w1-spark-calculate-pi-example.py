
import pyspark as ps
import random

spark = (ps.sql.SparkSession.builder
        .appName("get-pi")
        .getOrCreate()
        )
,
sc = spark.sparkContext
random.seed(1)

def sample(p):
    x, y = random.random(), random.random()
    return 1 if x*x + y*y < 1 else 0

count = (sc.parallelize(range(0, 10000000))
           .map(sample)
           .reduce(lambda a, b: a + b)
        )

result = {"pi": (4.0 * count / 10000000)}
print(result, file=open('calculate-pi-out.txt', 'w'))
