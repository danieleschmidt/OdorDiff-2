"""
Production Gunicorn Configuration for OdorDiff-2
High-performance WSGI server configuration with monitoring and security
"""

import multiprocessing
import os
from distutils.util import strtobool

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = int(os.environ.get('WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = os.environ.get('WORKER_CLASS', 'gevent')
worker_connections = int(os.environ.get('WORKER_CONNECTIONS', 1000))
max_requests = int(os.environ.get('MAX_REQUESTS', 1000))
max_requests_jitter = int(os.environ.get('MAX_REQUESTS_JITTER', 50))
preload_app = strtobool(os.environ.get('PRELOAD', 'True'))

# Timeout settings
timeout = int(os.environ.get('TIMEOUT', 30))
keepalive = int(os.environ.get('KEEPALIVE', 5))
graceful_timeout = int(os.environ.get('GRACEFUL_TIMEOUT', 30))

# Performance tuning
worker_tmp_dir = '/dev/shm'  # Use shared memory for better performance
worker_recycling = True
disable_redirect_access_to_syslog = True

# Security
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190

# Process naming
proc_name = 'odordiff2-api'
default_proc_name = 'odordiff2'

# Logging
log_level = os.environ.get('LOG_LEVEL', 'info').lower()
accesslog = '-'  # Log to stdout
errorlog = '-'   # Log to stderr
access_log_format = os.environ.get('ACCESS_LOG_FORMAT', 
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s %(M)s')

# Monitoring hooks
def when_ready(server):
    """Called just after the server is started."""
    server.log.info("OdorDiff-2 API server is ready. Listening on: %s", server.address)

def worker_int(worker):
    """Called just after a worker exited on SIGINT or SIGQUIT."""
    worker.log.info("Worker received INT or QUIT signal")

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_worker_init(worker):
    """Called just after a worker has initialized the application."""
    worker.log.info("Worker initialized (pid: %s)", worker.pid)

def worker_abort(worker):
    """Called when a worker received the SIGABRT signal."""
    worker.log.info("Worker received SIGABRT signal")

def pre_exec(server):
    """Called just before a new master process is forked."""
    server.log.info("Forked child, re-executing.")

def pre_request(worker, req):
    """Called just before a worker processes the request."""
    worker.log.debug("%s %s" % (req.method, req.path))

def post_request(worker, req, environ, resp):
    """Called after a worker processes the request."""
    pass

def child_exit(server, worker):
    """Called just after a worker has been reaped."""
    server.log.info("Worker exited (pid: %s)", worker.pid)

def worker_exit(server, worker):
    """Called just after a worker has been reaped."""
    server.log.info("Worker exited (pid: %s)", worker.pid)

def nworkers_changed(server, new_value, old_value):
    """Called just after num_workers has been changed."""
    server.log.info("Number of workers changed from %s to %s", old_value, new_value)

def on_exit(server):
    """Called just before exiting."""
    server.log.info("Shutting down: Master")

def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info("Starting OdorDiff-2 API server")

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    server.log.info("Reloading OdorDiff-2 API server")

# SSL Configuration (if certificates are available)
import ssl
cert_file = os.environ.get('SSL_CERT_FILE')
key_file = os.environ.get('SSL_KEY_FILE')
if cert_file and key_file:
    keyfile = key_file
    certfile = cert_file
    ssl_version = ssl.PROTOCOL_TLS
    cert_reqs = ssl.CERT_NONE
    ca_certs = None
    suppress_ragged_eofs = True
    ciphers = 'ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS'