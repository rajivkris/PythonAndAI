import pandas as pd
import random
from datetime import datetime, timedelta

teams = ['Backend Team', 'Frontend Team', 'Database Team', 'DevOps Team', 'Security Team']

log_messages = {
    'Backend Team': [
        'NullPointerException at {} in {}.java: line {}',
        'ArrayIndexOutOfBoundsException at {} in {}.java: line {}',
        'IllegalStateException: {} in {}.java: line {}',
        'NoSuchMethodError: {} in {}.java: line {}',
        'ClassCastException: {} in {}.java: line {}'
    ],
    'Frontend Team': [
        'React component failed to render at {}. Error: {}',
        'TypeError: Cannot read property {} of undefined at {}',
        'Unhandled Rejection ({}): {} in component {}',
        'ReferenceError: {} is not defined in {} at line {}',
        'SyntaxError: Unexpected token {} in JSON at position {}'
    ],
    'Database Team': [
        'SQLException: Could not connect to database {}. Error: {}',
        'DataIntegrityViolationException: {} at {}.java: line {}',
        'TransactionTimedOutException: Transaction {} timed out after {} ms',
        'JDBCConnectionException: {} at {}.java: line {}',
        'QueryTimeoutException: Query {} timed out after {} ms'
    ],
    'DevOps Team': [
        'Deployment failed for service {}. Error: {}',
        'Kubernetes pod {} failed to start. Reason: {}',
        'Docker container {} stopped unexpectedly. Error: {}',
        'CI/CD pipeline failed at stage {}. Error: {}',
        'Service {} is down. Reason: {}'
    ],
    'Security Team': [
        'Unauthorized access attempt detected from IP {}',
        'CSRF attack detected in {}.java at line {}',
        'SQL injection attempt blocked. Query: {}',
        'XSS attack detected in {} at {}',
        'Failed login attempt for user {} from IP {}'
    ]
}

# Define possible reasons for errors based on historical issues
historical_issues = {
    'Backend Team': [
        'Incorrect handling of null values',
        'Array index out of bounds',
        'Invalid state due to incorrect logic',
        'Missing method in class',
        'Improper type casting'
    ],
    'Frontend Team': [
        'Incorrect component state',
        'Undefined property access',
        'Unhandled promise rejection',
        'Undefined variable',
        'JSON parsing error'
    ],
    'Database Team': [
        'Database connection issues',
        'Data integrity constraints',
        'Transaction timeouts',
        'JDBC driver issues',
        'Query execution timeouts'
    ],
    'DevOps Team': [
        'Deployment configuration errors',
        'Kubernetes misconfigurations',
        'Docker container issues',
        'CI/CD pipeline misconfigurations',
        'Service health check failures'
    ],
    'Security Team': [
        'Unauthorized access attempts',
        'Cross-Site Request Forgery (CSRF)',
        'SQL injection attempts',
        'Cross-Site Scripting (XSS)',
        'Brute force login attempts'
    ]
}

def generate_log_data(num_logs):
    data = []
    start_date = datetime.now() - timedelta(days=30)  # Generate logs for the past 30 days
    for _ in range(num_logs):
        team = random.choice(teams)
        message_template = random.choice(log_messages[team])
        # Generate random values for placeholders
        placeholders_values = [
            'serviceA', 'serviceB', 'serviceC', random.randint(1000, 9999),
            'componentX', 'componentY', 'UserA', 'UserB',
            'database1', 'database2', 'IP address ' + random_ip()
        ]
        list_of_randon_values = random.choices(placeholders_values, k=message_template.count("{}"))
        message = message_template.format(*list_of_randon_values)
        timestamp = start_date + timedelta(minutes=random.randint(0, 43200))  # Random timestamp in the past 30 days
        reason = random.choice(historical_issues[team])
        data.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'log_message': message,
            'label': team,
            'possible_reason': reason
        })
    return data

def random_ip():
    return '.'.join(map(str, (random.randint(0, 255) for _ in range(4))))

num_logs = 30000
log_data = generate_log_data(num_logs)
df = pd.DataFrame(log_data)

df.to_csv('spring_boot_synthetic_log_data.csv', index=False)
print('Synthetic log data generated and saved to spring_boot_synthetic_log_data.csv')