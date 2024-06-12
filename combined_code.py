import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class WebVulnerabilityScanner:
    def __init__(self, target_url):
        self.target_url = target_url
        self.session = requests.Session()
        self.visited_links = set()
        self.vulnerabilities = []
        self.xss_payloads = self.load_xss_payloads()
        self.sql_injection_payloads = self.load_sql_injection_payloads()

    def load_xss_payloads(self):
        with open('xss-payload-list.txt', 'r', encoding='utf-8') as file:
            return [line.strip() for line in file.readlines()]

    def load_sql_injection_payloads(self):
        response = requests.get(
            "https://raw.githubusercontent.com/payloadbox/sql-injection-payload-list/master/README.md")
        sql_payloads = response.text.split("\n")
        sql_payloads = [payload.strip() for payload in sql_payloads if payload.strip() and not payload.startswith("#")]
        return sql_payloads

    def crawl(self, url):
        if url in self.visited_links:
            return
        print("[+] Crawling:", url)
        try:
            if url.startswith('http://') or url.startswith('https://'):
                response = self.session.get(url)
                if response.status_code == 200:
                    self.visited_links.add(url)
                    links = self.extract_links(response.text, url)
                    with ThreadPoolExecutor(max_workers=10) as executor:
                        executor.map(self.crawl, links)
                        executor.map(self.scan_page, links)
        except Exception as e:
            print("[-] Error crawling {}: {}".format(url, e))

    def extract_links(self, html_content, base_url):
        links = []
        soup = BeautifulSoup(html_content, 'html.parser')
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if not href:
                continue
            absolute_url = urljoin(base_url, href)
            links.append(absolute_url)
        return links

    def scan_page(self, url):
        print("[+] Scanning:", url)
        try:
            response = self.session.get(url)
            if response.status_code == 200:
                self.detect_xss_vulnerability(url, response)
                self.detect_sql_injection_vulnerability(url, response)
        except Exception as e:
            print("[-] Error scanning {}: {}".format(url, e))

    def detect_xss_vulnerability(self, url, response):
        forms = self.extract_forms(response.text)
        for form in forms:
            form_data = self.extract_form_data(form)
            for field_name, field_value in form_data.items():
                tampered_data = self.modify_field_value(field_value)
                tampered_url = self.build_tampered_url(url, form_data, {field_name: tampered_data})
                tampered_response = self.session.get(tampered_url)
                if self.is_xss_detected(tampered_response):
                    self.vulnerabilities.append({
                        "vulnerability": "XSS",
                        "url": tampered_url,
                        "form_data": form_data,
                        "payload": tampered_data
                    })

    def detect_sql_injection_vulnerability(self, url, response):
        for payload in self.sql_injection_payloads:
            tampered_url = url + payload
            tampered_response = self.session.get(tampered_url)
            if self.is_sql_injection_detected(tampered_response):
                self.vulnerabilities.append({
                    "vulnerability": "SQL Injection",
                    "url": tampered_url,
                    "payload": payload
                })

    def is_sql_injection_detected(self, response):
        sql_error_messages = [
            "SQL syntax",
            "MySQL server",
            "Syntax error",
            "Unclosed quotation mark",
            "You have an error in your SQL syntax",
            "Database error",
            "Microsoft SQL Server",
            "ODBC SQL",
            "PostgreSQL query failed",
            "Warning: mysql_fetch_array()",
            "Warning: mysql_fetch_assoc()",
            "Fatal error",
            "MySqlException",
            "PL/SQL",
            "PG::SyntaxError:",
            "ORA-00933:",
            "SQLiteException",
            "JDBCException",
            "SQLException"
        ]

        for error_message in sql_error_messages:
            if error_message.lower() in response.text.lower():
                return True

        return False

    def extract_forms(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.find_all('form')

    def extract_form_data(self, form):
        form_data = {}
        for input_field in form.find_all('input'):
            if input_field.get('name'):
                form_data[input_field['name']] = input_field.get('value', '')
        return form_data

    def modify_field_value(self, value):
        return self.xss_payloads.pop(0) if self.xss_payloads else '"><script>alert("XSS")</script>'

    def build_tampered_url(self, url, form_data, params):
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        query_params.update(params)
        modified_query = urlencode(query_params, doseq=True)
        tampered_url = urlunparse(parsed_url._replace(query=modified_query))
        return tampered_url

    def is_xss_detected(self, response):
        return re.search(r'<script>alert\("XSS"\)</script>', response.text, re.IGNORECASE)

    def generate_report(self):
        if self.vulnerabilities:
            print("[+] Vulnerabilities Found:")
            for vuln in self.vulnerabilities:
                print("    - Type:", vuln["vulnerability"])
                print("      URL:", vuln["url"])
                print("      Form Data:", vuln["form_data"])
                print("      Payload:", vuln["payload"])
                print()
        else:
            print("[+] No vulnerabilities found.")

    def scan_site(self):
        self.crawl(self.target_url)
        print("[+] Starting vulnerability scan...")
        self.generate_report()

# Read the text data
with open('firewall.txt', 'r') as file:
    data = file.readlines()

# Preprocess the text
processed_data = [text.lower() for text in data[:1000]]
y = np.random.randint(2, size=len(processed_data))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(processed_data, y, test_size=0.2, random_state=42)

# Vectorize the text data
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)

# Predict on the test set
predictions = rf_classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Combine training and testing data for prediction
combined_data = X_train + X_test

# Preprocess the combined data
processed_combined_data = [text.lower() for text in combined_data]

# Vectorize the combined data using the same TF-IDF vectorizer
X_combined_tfidf = tfidf_vectorizer.transform(processed_combined_data)

# Predict vulnerabilities using the trained classifier
predicted_labels_combined = rf_classifier.predict(X_combined_tfidf)

# Print the predicted labels for combined data
print("Predicted labels for combined data:")
for text, label in zip(combined_data, predicted_labels_combined):
    if label == 1:
        print(f"Vulnerability Detected: {text}")
    else:
        print(f"No Vulnerability Detected: {text}")

# Start the web vulnerability scan
target_url = input("Enter target URL: ")
scanner = WebVulnerabilityScanner(target_url)
scanner.scan_site()
