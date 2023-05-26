# All-Pairs-Similarity
Assignment III LwMD

- Make sure to add local source files as dependencies of the SparkContext; they must be sent to worker nodes, which are not in the drivers' (this) working directory.
    - Make sure to **compress and add the whole source folder** (e.g. `src/`) **as a `.zip` archive**, so that the package structure is preserved.
    - Example:
# Add local dependencies (local python source files) to SparkContext and sys.path
src_zip_path = os.path.abspath("../../src.zip")
spark.sparkContext.addPyFile(src_zip_path)
sys.path.insert(0, SparkFiles.getRootDirectory())

- Start the master and worker nodes by running `bash $SPARK_HOME/sbin/start-all.sh`
- Stop the master and worker nodes by running `bash $SPARK_HOME/sbin/stop-all.sh`