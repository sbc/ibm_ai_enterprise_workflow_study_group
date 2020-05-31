/usr/local/bin/spark/spark-submit \
--master local[*] \
--executor-memory 1G \
--driver-memory 1G \
$@
