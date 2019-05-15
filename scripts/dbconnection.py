#!/usr/bin/env python
"""dbconnection.py: Code to handle postgres db connection"""

__author__ = "Suhail Rehman"

import psycopg2


# Requires a ~/.pgpass file
def connect_db(hostname='localhost', dbname='lineage', user='lineage'):
    conn = psycopg2.connect(f"host={hostname} dbname={dbname} user={user}")
    return conn


def execute_insert(query, params, conn):
    try:
        conn.cursor().execute(query, params)
        conn.commit()
        return True
    except Exception as e:
        print(e)
        return False


def execute_query(query, params, conn):
    try:
        cur = conn.cursor()
        cur.execute(query, params)
        conn.commit()
        # print(cur.query)
        return cur.fetchall()
    except Exception as e:
        raise(e)
        return False


def test_db():
    conn = connect_db()
    curs = conn.cursor()
    res = curs.execute('select * from lineage.workflow;')
    print(res)


def truncate_db(connection):
    curs = connection.cursor()
    curs.execute('SELECT truncate_tables();')
    connection.commit()
