#!/usr/bin/env python
import pika

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='hello')

try:
    while True:
        msg_body = input('>')
        channel.basic_publish(exchange='',
                            routing_key='hello',
                            body=msg_body)
        print(f" [x] Sent {msg_body}")
except KeyboardInterrupt:
    print("Closing connection.")

connection.close()


# FORMAT MESSAGES LIKE THIS: <x>,<y>,<heading> (i.e., 1.0,2.0,3.0)
# Ranges are as follows: pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)
# Ranges are defined on line 106 in /IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/navigation/config/anymal_c/navigation_env_cfg.py
