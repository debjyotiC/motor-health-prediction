from edgemodelkit import DataFetcher

# Initialize for HTTP communication
fetcher = DataFetcher(source="serial", serial_port="/dev/cu.usbmodem1101", baud_rate=115200)

# Log 10 samples with timestamp and count columns
fetcher.log_sensor_data(class_label="MotorOn", num_samples=5000, add_timestamp=True, add_count=False)

