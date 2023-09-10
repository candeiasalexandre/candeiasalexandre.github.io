---
title           : "Exchanging AVRO messages between confluent_kafka and PySpark"
description     : "Exchanging AVRO messages between confluent_kafka and PySpark"
katex           : true
date: 2023-09-10
katexExtensions : [ mhchem, copy-tex ]
---

A common pattern to exchange data between (offline) batch jobs and micro-services is to use Kafka as the communication layer.
The data is wrapped in a message with a specific contract and is written to a Kafka topic so that it can later be consumed. The message needs to be serialized in a way that both the producer and consumer can understand, a way of specifying both the message contract (schema) and the serialization format is using the [AVRO](https://avro.apache.org/docs/1.10.2/spec.html) specification.

A AVRO schema is a json document that says which fields the message contains and the type of those fields. Take for example the following schema:

``` json
"type": "record",
"name": "ExampleMessage",
"namespace": "example.avro",
"fields": [
    {
        "name": "field_1",
        "type": "string"
    },
    {
        "name": "field_2",
        "type": "long"
    },
    {
        "name": "field_3",
        "type": {
            "type": "array",
            "items": "long"
        }
    }
]
```

This contract says that our messages will have a string, a (long) integer and an array of (long) integers named `field_1`, `field_2` and `field_3`. By using this contract both the producer and the consumer will know what to expect regarding the content of each message.

The contract is also important when transforming the data into bytes that can be transmitted over the network or keep in storage by Kafka in a efficient way. Those bytes might also need to be read back into memory and transformed in a representation that your programs understand. This whole process of SerDe (Serialization/Deserialization) is easily done using your favourite implementation of the AVRO specification.

[Spark](https://spark.apache.org/) is the de facto batch job library used across the industry. Since AVRO and Spark are widely used it is not surprising that Spark has native support to [serialize](https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.sql.avro.functions.to_avro.html?highlight=to_avro#pyspark.sql.avro.functions.to_avro) and [deserialize](https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.sql.avro.functions.from_avro.html?highlight=from_avro#pyspark.sql.avro.functions.from_avro) AVRO messages given a AVRO schema. Kafka is also supported by Spark which makes writing/reading data easy, check this [example](https://spark.apache.org/docs/latest/structured-streaming-kafka-integration.html#reading-data-from-kafka).

The question is **how well does the connection between Spark, AVRO and Kafka work within the rest of the python ecosystem** ?

To test it, we created 2 simple scenarios:
1. Producer written in PySpark. Consumer written in python using [confluent-kafka-python](https://github.com/confluentinc/confluent-kafka-python) client
2. Producer in python using [confluent-kafka-python](https://github.com/confluentinc/confluent-kafka-python) client. Consumer written in PySpark.

(All the code used in these tests can be found in this [repo](https://github.com/candeiasalexandre/pyspark_confluent_avro))

## Scenario 1: PySpark Producer & confluent-kafka-python Consumer

In this scenario, we create a column in a Spark DataFrame that contains structs with the same schema as showed before. The Serialization of this column to AVRO is done using the `to_avro` function provided by Spark. The messages are written to Kafka using Spark and a `confluent_kafka` consumer is used to read the messages back from Kafka.
You can see bellow the whole test, if you want more details you can check the code [here](https://github.com/candeiasalexandre/pyspark_confluent_avro/blob/9449174622b4ba5fcc026c23358fa8d2f5732d5d/tests/test_confluent_spark.py#L65).

``` python
@pytest.fixture()
def example_data(spark_session: SparkSession) -> DataFrame:
    num_rows = 10
    pdf = pd.DataFrame(
        {
            "id": list(range(num_rows)),
            "field_1": [f"field_1_{value}" for value in range(num_rows)],
            "field_2": list(range(num_rows)),
            "field_3": [
                [i for i in range(10, 10 + row_number)]
                for row_number in range(num_rows)
            ],
        }
    )

    df = spark_session.createDataFrame(pdf)
    df = df.withColumn("message", spark_func.struct("field_1", "field_2", "field_3"))

    return df

def test_write_spark_read_confluent(
    kafka_topic: KafkaOptions,
    schema_registry_client: SchemaRegistryClient,
    example_schema: Dict[str, Any],
    example_data: DataFrame,
) -> None:
    """
    This test shows that if we write a message to Kafka serialized in avro
    using the pyspark.sql.avro.functions.to_avro function,  we cannot read it in using the confluent_kafka consumer.
    We will get a deserialization error!
    """
    schema_json = json.dumps(example_schema)
    kafka_conf = {
        "bootstrap.servers": kafka_topic.host,
        "compression.type": kafka_topic.compression_type,
        "group.id": "pytest-tests",
        "auto.offset.reset": "earliest",
    }

    df_message_avro = example_data.withColumn(
        "message_avro",
        spark_to_avro(example_data["message"], schema_json),
    )

    write_spark_kafka(df_message_avro, "message_avro", kafka_topic)

    with pytest.raises(SerializationError) as e_info:
        _ = read_avro_confluent_kafka(
            kafka_conf,
            kafka_topic.topic,
            schema_registry_client,
            schema_json,
        )

    assert (
        " This message was not produced with a Confluent Schema Registry serializer"
        in str(e_info)
    )
```

When reading the messages back from Kafka using a `confluent_kafka` consumer, we see that we get a `SerializationError` saying that the message was not produced using a `conluent_kafka` serializer. This happens due to the consumer deserializer expecting a field, containing the schema version.

## Scenario 2: confluent-kafka-python Producer & PySpark Consumer

In this scenario, we produce messages using a `confluent_kafka` producer and we try to read them with a Spark consumer. The deserialization is done by using the `from_avro` Spark function.

``` python
@pytest.fixture()
def example_messages() -> List[Dict[str, Any]]:
    num_messages = 10
    return [
        {
            "id": int(msg_num),
            "field_1": f"field_1_{msg_num}",
            "field_2": msg_num,
            "field_3": [i for i in range(10, 10 + msg_num)],
        }
        for msg_num in range(num_messages)
    ]

def test_write_confluent_read_spark(
    kafka_topic: KafkaOptions,
    schema_registry_client: SchemaRegistryClient,
    example_schema: Dict[str, Any],
    example_messages: List[Dict[str, Any]],
    spark_session: SparkSession,
) -> None:
    """
    This test shows that if we write a message to Kafka serialized in avro
    using the the confluent_kafka producer, we cannot deserialize it correctly using the
    pyspark.sql.avro.functions.from_avro.
    We will messages populate with default fields, and not the correct data.
    """
    kafka_conf = {
        "bootstrap.servers": kafka_topic.host,
        "compression.type": kafka_topic.compression_type,
        "group.id": "pytest-tests",
        "auto.offset.reset": "earliest",
    }

    schema_json = json.dumps(example_schema)

    write_avro_confluent_kafka(
        example_messages,
        kafka_conf,
        kafka_topic.topic,
        schema_registry_client,
        schema_json,
    )

    df_messages_read = read_spark_kafka(spark_session, kafka_topic)
    pdf_messages_read = df_messages_read.withColumn(
        "message",
        spark_from_avro(df_messages_read["value"], schema_json),
    ).toPandas()

    field_1_read_messages = set(
        [x["field_1"] for x in pdf_messages_read["message"].to_list()]
    )
    field_1_original_messages = set([x["field_1"] for x in example_messages])

    assert field_1_original_messages.intersection(field_1_read_messages) == set()
```

(You can find the code for this test [here](https://github.com/candeiasalexandre/pyspark_confluent_avro/blob/9449174622b4ba5fcc026c23358fa8d2f5732d5d/tests/test_confluent_spark.py#L105).)

When reading the messages using Spark, we see a weird behaviour. We don't get any deserialization errors but **the information present in the read messages is not correct**. For example, in the test above we see that the overlap of the values written and read is empty for `field_1` .

## Solution

From the two scenarios above we might think that there is no easy way to exchange data between PySpark and [confluent-kafka-python](https://github.com/confluentinc/confluent-kafka-python) making us think that a custom solution might be needed. If we think, the whole issue is related with the SerDe implementation expected by [confluent-kafka-python](https://github.com/confluentinc/confluent-kafka-python) so we need to work at the level of the `from_avro`, `to_avro` implementations given by Spark to make it compliant with what `confluent_kafka` expects.

Thankfully, someone already did the heavy lifting for us and we can use the custom [ABRis](https://github.com/AbsaOSS/ABRiS/) SerDe functions. This package is only available in Scala, but we can also make it work in PySpark by using the Py4j connection present in the `SparkContext` and adding some configuration to our [`SparkSession`](https://github.com/candeiasalexandre/pyspark_confluent_avro/blob/5cc483d8e5e95363e2802d80c16a053225eeefbe/tests/conftest.py#L103). We can create our custom [`from_avro`](https://github.com/candeiasalexandre/pyspark_confluent_avro/blob/fcd8d7019b0b111d167e19be9515b080ea5e0c73/pyspark_confluent_avro/spark_avro_serde.py#L28) and [`to_avro`](https://github.com/candeiasalexandre/pyspark_confluent_avro/blob/fcd8d7019b0b111d167e19be9515b080ea5e0c73/pyspark_confluent_avro/spark_avro_serde.py#L59) functions that can be used to replace the Spark specific ones.

Using these new functions, in scenario 1 we are capable of reading the messages written by Spark using `confluent_kafka` without `SerializationError` issues.

``` python
def test_write_spark_custom_read_confluent(
    kafka_topic: KafkaOptions,
    schema_registry_client: SchemaRegistryClient,
    example_schema: Dict[str, Any],
    example_data: DataFrame,
    schema_registry_config: Dict[str, str],
    example_messages: List[Dict[str, Any]],
) -> None:
    """
    Test that shows correctness if we use the Abris package to write an Avro serialized message.
    If we serialize the messages with Abris, when we read from kafka using kafka_confluent we
    get the right data.
    """
    schema_json = json.dumps(example_schema)
    kafka_conf = {
        "bootstrap.servers": kafka_topic.host,
        "compression.type": kafka_topic.compression_type,
        "group.id": "pytest-tests",
        "auto.offset.reset": "earliest",
    }

    # this will register the schema automatically
    # check https://github.com/AbsaOSS/ABRiS
    abris_config = to_avro_abris_config(
        {"schema.registry.url": schema_registry_config["url"]},
        kafka_topic.topic,
        False,
        schema_json=schema_json,
    )
    df_message_avro = example_data.withColumn(
        "message_avro",
        custom_to_avro(example_data["message"], abris_config),
    )
    write_spark_kafka(df_message_avro, "message_avro", kafka_topic)

    read_messages = read_avro_confluent_kafka(
        kafka_conf,
        kafka_topic.topic,
        schema_registry_client,
        schema_json,
    )

    field_1_read_messages = set([x["field_1"] for x in read_messages])
    field_1_original_messages = set([x["field_1"] for x in example_messages])

    assert len(read_messages) == len(example_messages)
    assert field_1_read_messages == field_1_original_messages
```

(you can find the code of this snippet [here](https://github.com/candeiasalexandre/pyspark_confluent_avro/blob/9449174622b4ba5fcc026c23358fa8d2f5732d5d/tests/test_confluent_spark.py#L197))

The same for scenario 2, the messages read into the Spark DataFrame have now the correct information that was written by the `confluent_kafka`.

``` python
def test_write_confluent_read_spark_custom(
    kafka_topic: KafkaOptions,
    schema_registry_client: SchemaRegistryClient,
    example_schema: Dict[str, Any],
    example_messages: List[Dict[str, Any]],
    spark_session: SparkSession,
    schema_registry_config: Dict[str, str],
) -> None:
    """
    Test that shows correctness if we use the Abris package to read a confluent Avro serialized message.
    IF we use Abris from_avro, we get the correct data when reading.
    """
    kafka_conf = {
        "bootstrap.servers": kafka_topic.host,
        "compression.type": kafka_topic.compression_type,
        "group.id": "pytest-tests",
        "auto.offset.reset": "earliest",
    }
    schema_json = json.dumps(example_schema)

    write_avro_confluent_kafka(
        example_messages,
        kafka_conf,
        kafka_topic.topic,
        schema_registry_client,
        schema_json,
    )

    df_messages_read = read_spark_kafka(spark_session, kafka_topic)
    abris_config = from_avro_abris_config(
        {"schema.registry.url": schema_registry_config["url"]}, kafka_topic.topic, False
    )
    pdf_messages_read = df_messages_read.withColumn(
        "message",
        custom_from_avro(df_messages_read["value"], abris_config),
    ).toPandas()

    field_1_read_messages = set(
        [x["field_1"] for x in pdf_messages_read["message"].to_list()]
    )
    field_1_original_messages = set([x["field_1"] for x in example_messages])

    assert (
        field_1_original_messages.intersection(field_1_read_messages)
        == field_1_original_messages
    )
```

(you can find the code of this snippet [here](https://github.com/candeiasalexandre/pyspark_confluent_avro/blob/9449174622b4ba5fcc026c23358fa8d2f5732d5d/tests/test_confluent_spark.py#L149))

An alternative to using ABRis would be to implement extra logic to SerDe the messages using custom python UDF's. While possible (and probably resulting in more pythonic or cleaner code), by using python UDF's you are usually adding an overhead due to the python-jvm communication making your workloads run slower.

To conclude, I hope that this post saves you some headache the next time you have to exchange data between `confluent_kafka` and PySpark. If you feel that a lot of information in this post is missing such as the specific implementations of the functions you see in the code snippets, you can check this [repo](https://github.com/candeiasalexandre/pyspark_confluent_avro) for more details.

Thanks for reading :)
