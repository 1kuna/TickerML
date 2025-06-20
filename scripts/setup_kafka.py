#!/usr/bin/env python3
"""
Kafka Setup and Deployment Script
Sets up single-node Kafka cluster for TickerML trading system
"""

import os
import sys
import subprocess
import time
import yaml
import logging
from pathlib import Path
import requests
import tarfile
import platform

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KafkaSetup:
    """Setup and manage Kafka cluster"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.kafka_dir = self.project_root / "infrastructure" / "kafka"
        self.kafka_version = "2.13-3.6.0"
        self.kafka_url = f"https://downloads.apache.org/kafka/3.6.0/kafka_{self.kafka_version}.tgz"
        
        # Load configuration
        config_path = self.project_root / "config" / "kafka_config.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.kafka_home = self.kafka_dir / f"kafka_{self.kafka_version}"
        
    def check_java(self):
        """Check if Java is available"""
        try:
            result = subprocess.run(['java', '-version'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Java is available")
                return True
            else:
                logger.error("Java not found. Please install Java 8 or higher")
                return False
        except FileNotFoundError:
            logger.error("Java not found. Please install Java 8 or higher")
            return False
    
    def download_kafka(self):
        """Download Kafka if not exists"""
        kafka_archive = self.kafka_dir / f"kafka_{self.kafka_version}.tgz"
        
        if self.kafka_home.exists():
            logger.info("Kafka already downloaded")
            return True
        
        # Create directory
        self.kafka_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading Kafka {self.kafka_version}...")
        try:
            response = requests.get(self.kafka_url, stream=True)
            response.raise_for_status()
            
            with open(kafka_archive, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info("Extracting Kafka...")
            with tarfile.open(kafka_archive, 'r:gz') as tar:
                tar.extractall(self.kafka_dir)
            
            # Remove archive
            kafka_archive.unlink()
            
            logger.info("Kafka downloaded and extracted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading Kafka: {e}")
            return False
    
    def create_kafka_config(self):
        """Create Kafka server configuration"""
        config_dir = self.kafka_home / "config"
        
        # Server properties
        server_props = config_dir / "server.properties"
        
        kafka_config = f"""
# Broker configuration
broker.id=0
listeners=PLAINTEXT://localhost:9092
advertised.listeners=PLAINTEXT://localhost:9092
num.network.threads=8
num.io.threads=8
socket.send.buffer.bytes=102400
socket.receive.buffer.bytes=102400
socket.request.max.bytes=104857600

# Log configuration
log.dirs={self.kafka_dir}/kafka-logs
num.partitions=3
num.recovery.threads.per.data.dir=1
offsets.retention.check.interval.ms=600000
offsets.retention.minutes=10080
transaction.state.log.replication.factor=1
transaction.state.log.min.isr=1
log.retention.hours=168
log.retention.check.interval.ms=300000
log.segment.bytes=1073741824
log.cleanup.policy=delete

# Zookeeper
zookeeper.connect=localhost:2181
zookeeper.connection.timeout.ms=18000

# Performance tuning for home use
replica.fetch.max.bytes=1048576
message.max.bytes=1048576
replica.socket.timeout.ms=30000
replica.socket.receive.buffer.bytes=65536
"""
        
        with open(server_props, 'w') as f:
            f.write(kafka_config.strip())
        
        logger.info("Created Kafka server configuration")
        
        # ZooKeeper properties (using embedded ZooKeeper)
        zk_props = config_dir / "zookeeper.properties"
        zk_config = f"""
dataDir={self.kafka_dir}/zookeeper
clientPort=2181
maxClientCnxns=0
admin.enableServer=false
"""
        
        with open(zk_props, 'w') as f:
            f.write(zk_config.strip())
        
        logger.info("Created ZooKeeper configuration")
    
    def start_zookeeper(self):
        """Start ZooKeeper server"""
        logger.info("Starting ZooKeeper...")
        
        zk_script = self.kafka_home / "bin" / "zookeeper-server-start.sh"
        if platform.system() == "Windows":
            zk_script = self.kafka_home / "bin" / "windows" / "zookeeper-server-start.bat"
        
        zk_config = self.kafka_home / "config" / "zookeeper.properties"
        
        # Make script executable
        if platform.system() != "Windows":
            os.chmod(zk_script, 0o755)
        
        # Start ZooKeeper in background
        cmd = [str(zk_script), str(zk_config)]
        
        try:
            # Create log directory
            log_dir = self.kafka_dir / "logs"
            log_dir.mkdir(exist_ok=True)
            
            with open(log_dir / "zookeeper.log", 'w') as log_file:
                zk_process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=str(self.kafka_home)
                )
            
            # Wait for ZooKeeper to start
            time.sleep(10)
            
            if zk_process.poll() is None:
                logger.info("ZooKeeper started successfully")
                return zk_process
            else:
                logger.error("ZooKeeper failed to start")
                return None
                
        except Exception as e:
            logger.error(f"Error starting ZooKeeper: {e}")
            return None
    
    def start_kafka(self):
        """Start Kafka server"""
        logger.info("Starting Kafka server...")
        
        kafka_script = self.kafka_home / "bin" / "kafka-server-start.sh"
        if platform.system() == "Windows":
            kafka_script = self.kafka_home / "bin" / "windows" / "kafka-server-start.bat"
        
        kafka_config = self.kafka_home / "config" / "server.properties"
        
        # Make script executable
        if platform.system() != "Windows":
            os.chmod(kafka_script, 0o755)
        
        # Start Kafka in background
        cmd = [str(kafka_script), str(kafka_config)]
        
        try:
            log_dir = self.kafka_dir / "logs"
            log_dir.mkdir(exist_ok=True)
            
            with open(log_dir / "kafka.log", 'w') as log_file:
                kafka_process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=str(self.kafka_home)
                )
            
            # Wait for Kafka to start
            time.sleep(15)
            
            if kafka_process.poll() is None:
                logger.info("Kafka server started successfully")
                return kafka_process
            else:
                logger.error("Kafka server failed to start")
                return None
                
        except Exception as e:
            logger.error(f"Error starting Kafka server: {e}")
            return None
    
    def create_topics(self):
        """Create required Kafka topics"""
        logger.info("Creating Kafka topics...")
        
        topics_script = self.kafka_home / "bin" / "kafka-topics.sh"
        if platform.system() == "Windows":
            topics_script = self.kafka_home / "bin" / "windows" / "kafka-topics.bat"
        
        topics = self.config['kafka']['topics']
        
        for topic_key, topic_name in topics.items():
            try:
                cmd = [
                    str(topics_script),
                    '--create',
                    '--topic', topic_name,
                    '--bootstrap-server', 'localhost:9092',
                    '--partitions', '3',
                    '--replication-factor', '1'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"Created topic: {topic_name}")
                elif "already exists" in result.stderr:
                    logger.info(f"Topic already exists: {topic_name}")
                else:
                    logger.error(f"Error creating topic {topic_name}: {result.stderr}")
                    
            except Exception as e:
                logger.error(f"Error creating topic {topic_name}: {e}")
        
        # Create aggregate topics
        aggregate_topics = [
            f"{topics['trades']}-aggregates",
            f"{topics['news']}-sentiment",
            "crypto-features"  # Generated by feature consumer
        ]
        
        for topic_name in aggregate_topics:
            try:
                cmd = [
                    str(topics_script),
                    '--create',
                    '--topic', topic_name,
                    '--bootstrap-server', 'localhost:9092',
                    '--partitions', '3',
                    '--replication-factor', '1'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"Created aggregate topic: {topic_name}")
                elif "already exists" in result.stderr:
                    logger.info(f"Aggregate topic already exists: {topic_name}")
                else:
                    logger.error(f"Error creating aggregate topic {topic_name}: {result.stderr}")
                    
            except Exception as e:
                logger.error(f"Error creating aggregate topic {topic_name}: {e}")
    
    def test_kafka(self):
        """Test Kafka installation"""
        logger.info("Testing Kafka installation...")
        
        try:
            # Test producer
            from kafka import KafkaProducer
            producer = KafkaProducer(
                bootstrap_servers=['localhost:9092'],
                value_serializer=lambda v: str(v).encode('utf-8')
            )
            
            # Send test message
            producer.send('test-topic', 'Hello Kafka!')
            producer.flush()
            producer.close()
            
            logger.info("Kafka producer test successful")
            
            # Test consumer
            from kafka import KafkaConsumer
            consumer = KafkaConsumer(
                'test-topic',
                bootstrap_servers=['localhost:9092'],
                auto_offset_reset='earliest',
                consumer_timeout_ms=5000
            )
            
            messages = list(consumer)
            consumer.close()
            
            if messages:
                logger.info("Kafka consumer test successful")
                return True
            else:
                logger.warning("No messages received in consumer test")
                return False
                
        except Exception as e:
            logger.error(f"Kafka test failed: {e}")
            return False
    
    def create_systemd_services(self):
        """Create systemd services for Kafka (Linux only)"""
        if platform.system() != "Linux":
            logger.info("Systemd services only available on Linux")
            return
        
        logger.info("Creating systemd services...")
        
        # ZooKeeper service
        zk_service = f"""[Unit]
Description=Apache ZooKeeper server (TickerML)
Documentation=http://zookeeper.apache.org
Requires=network.target remote-fs.target
After=network.target remote-fs.target

[Service]
Type=simple
User={os.getenv('USER')}
ExecStart={self.kafka_home}/bin/zookeeper-server-start.sh {self.kafka_home}/config/zookeeper.properties
ExecStop={self.kafka_home}/bin/zookeeper-server-stop.sh
Restart=on-abnormal

[Install]
WantedBy=multi-user.target
"""
        
        # Kafka service
        kafka_service = f"""[Unit]
Description=Apache Kafka server (TickerML)
Documentation=http://kafka.apache.org/documentation.html
Requires=zookeeper-tickerml.service
After=zookeeper-tickerml.service

[Service]
Type=simple
User={os.getenv('USER')}
ExecStart={self.kafka_home}/bin/kafka-server-start.sh {self.kafka_home}/config/server.properties
ExecStop={self.kafka_home}/bin/kafka-server-stop.sh
Restart=on-abnormal

[Install]
WantedBy=multi-user.target
"""
        
        try:
            # Write service files
            with open('/tmp/zookeeper-tickerml.service', 'w') as f:
                f.write(zk_service)
            
            with open('/tmp/kafka-tickerml.service', 'w') as f:
                f.write(kafka_service)
            
            # Move to systemd directory (requires sudo)
            logger.info("Moving service files to /etc/systemd/system/")
            logger.info("You may need to run the following commands with sudo:")
            logger.info("sudo mv /tmp/zookeeper-tickerml.service /etc/systemd/system/")
            logger.info("sudo mv /tmp/kafka-tickerml.service /etc/systemd/system/")
            logger.info("sudo systemctl daemon-reload")
            logger.info("sudo systemctl enable zookeeper-tickerml kafka-tickerml")
            logger.info("sudo systemctl start zookeeper-tickerml kafka-tickerml")
            
        except Exception as e:
            logger.error(f"Error creating systemd services: {e}")
    
    def setup(self):
        """Complete Kafka setup"""
        logger.info("Starting Kafka setup for TickerML...")
        
        # Check prerequisites
        if not self.check_java():
            return False
        
        # Download and install Kafka
        if not self.download_kafka():
            return False
        
        # Create configuration
        self.create_kafka_config()
        
        # Start services
        zk_process = self.start_zookeeper()
        if not zk_process:
            return False
        
        kafka_process = self.start_kafka()
        if not kafka_process:
            return False
        
        # Create topics
        self.create_topics()
        
        # Test installation
        if self.test_kafka():
            logger.info("Kafka setup completed successfully!")
            
            # Create systemd services
            self.create_systemd_services()
            
            logger.info("\nKafka is now running:")
            logger.info("ZooKeeper: localhost:2181")
            logger.info("Kafka: localhost:9092")
            logger.info("\nTo stop Kafka:")
            logger.info(f"{self.kafka_home}/bin/kafka-server-stop.sh")
            logger.info(f"{self.kafka_home}/bin/zookeeper-server-stop.sh")
            
            return True
        else:
            logger.error("Kafka test failed")
            return False


def main():
    """Main entry point"""
    setup = KafkaSetup()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test-only":
        # Just test existing installation
        setup.test_kafka()
    else:
        # Full setup
        success = setup.setup()
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()