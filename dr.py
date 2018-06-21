import random
from enum import Enum
import pymysql
import pymysql.cursors
import json
import datetime as dt
import requests
import csv
import pprint as pp
import sys
import hashlib
from collections import defaultdict

class FIELDS(Enum):
    _1 = 0
    _2 = 1
    _3 = 2
    _4 = 3
    _5 = 4
    _6 = 5
    _7 = 6
    _8 = 7
    _9 = 8
    _10 = 9
    _11 = 10

class SQLConnector():
    def __init__(self, host='localhost',
                 pwd=None, db='db', user=None):
        self.connection = pymysql.connect(host=host,
                                          user=user,
                                          password=pwd,
                                          db=db,
                                          cursorclass=pymysql.cursors.DictCursor)

    def commit(self):
        self.connection.commit()
    def exec_sql(self, sql):
        with self.connection.cursor() as cursor:
            # Read a single record
            num_rows = cursor.execute(sql)
            result = []
            for i in range(num_rows):
                result.append(cursor.fetchone())
            return result
        return None




