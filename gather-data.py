from edgemodelkit import DataFetcher

# Initialize for HTTP communication
fetcher = DataFetcher(source="http", api_url="http://10.248.129.135")

# Log 10 samples with timestamp and count columns
fetcher.log_sensor_data(class_label="MotorOn", num_samples=50, add_timestamp=False, add_count=False)

