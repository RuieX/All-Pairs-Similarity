# All-Pairs-Similarity
Assignment III LwMD
./bin/pyspark from spark directory for spark shell

- Start the master and worker nodes by running `bash $SPARK_HOME/sbin/start-all.sh`
    - In case the following error happens: `localhost: ssh: connect to host localhost port 22: Connection refused`,
    - make sure to have the ssh daemon installed and running. If not installed, run `sudo apt install openssh-server`.
- Stop the master and worker nodes by running `bash $SPARK_HOME/sbin/stop-all.sh`