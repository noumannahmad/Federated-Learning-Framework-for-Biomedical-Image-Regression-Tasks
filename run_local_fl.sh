#!/bin/bash

# Launch FL server
echo "Starting Flower Server..."
python fl_simulation/server.py &

# Give the server some time to start
sleep 3

# Start simulated clients (in background)
echo "Starting Client 0..."
python fl_simulation/client.py --client_id=0 &

echo "Starting Client 1..."
python fl_simulation/client.py --client_id=1 &

# Wait for all background jobs to complete
wait
