

from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.rosbag1 import Writer

src = Path("UrbanNav-HK_Whampoa-20210521_sensors.bag")
dst = Path("UrbanNav-HK_Whampoa-20210521_no_camera.bag")

with AnyReader([src]) as reader, Writer(dst) as writer:
    connections = {}

    for connection in reader.connections:
        topic = connection.topic.lower()

        if "camera" in topic or "image" in topic:
            continue

        connections[connection.id] = writer.add_connection(
            connection.topic,
            connection.msgtype,
            typestore=reader.typestore,
        )

    for connection, timestamp, rawdata in reader.messages():
        new_connection = connections.get(connection.id)

        if new_connection is not None:
            writer.write(new_connection, timestamp, rawdata)