# Import dependencies
from jira import JIRA, JIRAError
from collections import Counter, defaultdict
from datetime import datetime
from time import sleep

import numpy as np
import pandas as pd
import networkx as nx

# Import dependencies

# Create instance for interacting with Jira
url = "jira.cloudera.com"
username = "gkshirsagar"
password = "HariharFort@123"
jira = JIRA(options={'server': url}, basic_auth=(username, password))

# Read data from Jira
try:
    jql = "project in ('DEX') AND created > '-365d'"
    # Search issues
    block_size = 100
    block_num = 0
    jira_search = jira.search_issues(jql, startAt=block_num*block_size, maxResults=block_size,
                                     fields="issuetype, created, resolutiondate, reporter, assignee, status, comments")

    # Define parameters for writing data
    index_beg = 0
    header = True
    mode = 'w'

    # Iteratively read data
    while bool(jira_search):
        # Container for Jira's data
        data_jira = []

        for issue in jira_search:
            # Get issue key
            issue_key = issue.key

            # Get request type
            request_type = str(issue.fields.issuetype)

            # Get datetime creation
            datetime_creation = issue.fields.created
            if datetime_creation is not None:
                # Interested in only seconds precision, so slice unnecessary part
                datetime_creation = datetime.strptime(datetime_creation[:19], "%Y-%m-%dT%H:%M:%S")

            # Get datetime resolution
            datetime_resolution = issue.fields.resolutiondate
            if datetime_resolution is not None:
                # Interested in only seconds precision, so slice unnecessary part
                datetime_resolution = datetime.strptime(datetime_resolution[:19], "%Y-%m-%dT%H:%M:%S")

            # Get reporter’s login and name
            reporter_login = None
            reporter_name = None
            reporter = issue.raw['fields'].get('reporter', None)
            if reporter is not None:
                reporter_login = reporter.get('key', None)
                reporter_name = reporter.get('displayName', None)

            # Get assignee’s login and name
            assignee_login = None
            assignee_name = None
            assignee = issue.raw['fields'].get('assignee', None)
            if assignee is not None:
                assignee_login = assignee.get('key', None)
                assignee_name = assignee.get('displayName', None)

            # Get status
            status = None
            st = issue.fields.status
            if st is not None:
                status = st.name

            # Get comments
            comments = None
            comments = issue.fields.comment.comments

            # Add data to data_jira
            data_jira.append((issue_key, request_type, datetime_creation, datetime_resolution, reporter_login, reporter_name, assignee_login, assignee_name, status, comments))

        # Write data read from Jira
        index_end = index_beg + len(data_jira)
        data_jira = pd.DataFrame(data_jira, index=range(index_beg, index_end),
                                     columns=['Issue key', 'Request type', 'Datetime creation', 'Datetime resolution', 'Reporter login', 'Reporter name', 'Assignee login', 'Assignee name', 'Status', 'Comments'])
        data_jira.to_csv(path_or_buf='data_jira.csv', sep=';', header=header, index=True, index_label='N', mode=mode)

        # Update for the next iteration
        block_num = block_num + 1
        index_beg = index_end
        header = False
        mode = 'a'

        # Print how many issues were read
        if block_num % 50 == 0:
            print(block_num * block_size)

        # Pause before next reading – it’s optional, just to be sure we will not overload Jira’s server
        sleep(1)

        # New issues search
        jira_search = jira.search_issues(jql, startAt=block_num*block_size, maxResults=block_size,
                                         fields="issuetype, created, resolutiondate, reporter, assignee, status")

    jira.close()
except (JIRAError, AttributeError):
    jira.close()
    print('Error')