# 1. TCP + ZMQ 双路发送（你当前的命令）
./OrinVideoSender --listen 192.168.200.112:13579 --zmq tcp://*:5555

# # 2. 只使用ZMQ发送
# ./OrinVideoSender --send --zmq tcp://*:5555

# 3. 只使用TCP发送（原来的方式）
# ./OrinVideoSender --listen 192.168.200.112:13579
