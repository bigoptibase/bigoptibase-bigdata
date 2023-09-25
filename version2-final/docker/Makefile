.PHONY: \
	build \
	clean \
	hadoop_base \
	hadoop_namenode \
	hadoop_datanode \
	hadoop_yarn_resourcemanager \
	hadoop_yarn_nodemanager \
	spark \
	zookeeper \
	hbase_base \
	hbase_master \
	hbase_regionserver \
	postgres \
	livy \
	bda \
	keycloak \
	kafka


build: \
	hadoop_base \
	hadoop_namenode \
	hadoop_datanode \
	hadoop_yarn_resourcemanager \
	hadoop_yarn_nodemanager \
	spark \
	zookeeper \
	hbase_base \
	hbase_master \
	hbase_regionserver \
	postgres \
	livy \
	bda \
	keycloak \
	kafka


include .env
export

hadoop_base:
	docker build \
		--build-arg HADOOP_VERSION=$(HADOOP_VERSION) \
		--tag bigoptibase/hadoop-base-$(HADOOP_VERSION) \
		./hadoop/base

hadoop_namenode: hadoop_base
	docker build \
		--build-arg HADOOP_VERSION=$(HADOOP_VERSION) \
		--tag bigoptibase/hadoop-namenode-$(HADOOP_VERSION) \
		./hadoop/namenode

hadoop_datanode: hadoop_base
	docker build \
		--build-arg HADOOP_VERSION=$(HADOOP_VERSION) \
		--tag bigoptibase/hadoop-datanode-$(HADOOP_VERSION) \
		./hadoop/datanode

hadoop_yarn_resourcemanager: hadoop_base 
	docker build \
		--build-arg HADOOP_VERSION=$(HADOOP_VERSION) \
		--tag bigoptibase/hadoop-yarn-resourcemanager-$(HADOOP_VERSION) \
		./hadoop/resourcemanager

hadoop_yarn_nodemanager: hadoop_base
	docker build \
		--build-arg HADOOP_VERSION=$(HADOOP_VERSION) \
		--tag bigoptibase/hadoop-yarn-nodemanager-$(HADOOP_VERSION) \
		./hadoop/nodemanager

spark: hadoop_yarn_nodemanager
	docker build \
		--build-arg HADOOP_VERSION=$(HADOOP_VERSION) \
		--build-arg SPARK_VERSION=$(SPARK_VERSION) \
                --build-arg HBASE_VERSION=$(HBASE_VERSION) \
		--tag bigoptibase/spark-$(SPARK_VERSION) \
		./spark

zookeeper: hadoop_base
	docker build \
		--build-arg HADOOP_VERSION=$(HADOOP_VERSION) \
		--build-arg ZOOKEEPER_VERSION=$(ZOOKEEPER_VERSION) \
		--tag bigoptibase/zookeeper-$(ZOOKEEPER_VERSION) \
		./zookeeper

hbase_base: hadoop_base
	docker build \
		--build-arg HADOOP_VERSION=$(HADOOP_VERSION) \
		--build-arg HBASE_VERSION=$(HBASE_VERSION) \
		--tag bigoptibase/hbase-base-$(HBASE_VERSION) \
		./hbase/base

hbase_master: hbase_base
	docker build \
		--build-arg HADOOP_VERSION=$(HADOOP_VERSION) \
		--build-arg HBASE_VERSION=$(HBASE_VERSION) \
		--tag bigoptibase/hbase-master-$(HBASE_VERSION) \
		./hbase/master

hbase_regionserver: hbase_base
	docker build \
		--build-arg HADOOP_VERSION=$(HADOOP_VERSION) \
		--build-arg HBASE_VERSION=$(HBASE_VERSION) \
		--tag bigoptibase/hbase-regionserver-$(HBASE_VERSION) \
		./hbase/regionserver

postgres:
	docker build \
		--tag bigoptibase/postgres-11 \
		./postgres

livy: spark
	docker build \
		--build-arg SPARK_VERSION=$(SPARK_VERSION) \
		--build-arg LIVY_VERSION=$(LIVY_VERSION) \
		--tag bigoptibase/livy-$(LIVY_VERSION) \
		./livy

bda: spark
	docker build \
		--build-arg SPARK_VERSION=$(SPARK_VERSION) \
		--file ./bda/Dockerfile \
		--tag bigoptibase/bda \
		./bda

keycloak: 
	docker build \
                --build-arg KEYCLOAK_VERSION=$(KEYCLOAK_VERSION) \
                --file ./keycloak/Dockerfile \
                --tag bigoptibase/keycloak \
                ./keycloak

kafka:
	docker build \
		--build-arg kafka_version=$(KAFKA_VERSION) \
		--file ./kafka/Dockerfile \
		--tag bigoptibase/kafka-$(KAFKA_VERSION) \
		./kafka


clean:
	docker rmi \
		bigoptibase/hadoop-base-$(HADOOP_VERSION) \
		bigoptibase/hadoop-namenode-$(HADOOP_VERSION) \
		bigoptibase/hadoop-datanode-$(HADOOP_VERSION) \
		bigoptibase/hadoop-yarn-resourcemanager-$(HADOOP_VERSION) \
		bigoptibase/hadoop-yarn-nodemanager-$(HADOOP_VERSION) \
		bigoptibase/spark-$(SPARK_VERSION) \
		bigoptibase/zookeeper-$(ZOOKEEPER_VERSION) \
		bigoptibase/hbase-base-$(HBASE_VERSION) \
		bigoptibase/hbase-master-$(HBASE_VERSION) \
		bigoptibase/hbase-regionserver-$(HBASE_VERSION) \
		bigoptibase/postgres-11 \
		bigoptibase/livy-$(LIVY_VERSION) \
		bigoptibase/bda \
		bigoptibase/keycloak \
                bigoptibase/kafka-$(KAFKA_VERSION)

