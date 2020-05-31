#!/usr/bin/env python

import os
import sys
import getopt
import scipy.stats as stats
import pandas as pd
import numpy as np
import sqlite3

DATA_DIR = os.path.join(".", "data")


def connect_db(file_path):
    """
    function to connection to aavail database
    """
    try:
        conn = sqlite3.connect(file_path)
        print("...successfully connected to db")
    except Error as e:
        print("...unsuccessful connection", e)

    return(conn)


def ingest_db_data(conn):
    """
    load and clean the db data
    """

    query = """
            SELECT cu.customer_id, cu.last_name, cu.first_name, cu.DOB,
            cu.city, cu.state, co.country_name, cu.gender
            FROM CUSTOMER cu
            INNER JOIN COUNTRY co
            ON cu.country_id = co.country_id;
            """
    _data = [d for d in conn.execute(query)]
    columns = ["customer_id", "last_name", "first_name",
               "DOB", "city", "state", "country", "gender"]
    df_db = pd.DataFrame(_data, columns=columns)
    duplicate_rows = df_db.duplicated()
    if True in duplicate_rows:
        df_db = df_db[~duplicate_rows]
        df_db.reset_index()
    print("... removed {} duplicate rows in db data".format(
        np.where(duplicate_rows == True)[0].size))
    return(df_db)


def ingest_stream_data(file_path):
    """
    load and clean the stream data
    """

    df_streams = pd.read_csv(file_path)
    customer_ids = df_streams['customer_id'].values
    unique_ids = np.unique(df_streams['customer_id'].values)
    streams = df_streams['subscription_stopped'].values
    has_churned = [0 if streams[customer_ids ==
                                uid].max() > 0 else 1 for uid in unique_ids]
    df_churn = pd.DataFrame(
        {"customer_id": unique_ids, "is_subscriber": has_churned})

    missing_stream_ids = np.isnan(df_streams['stream_id'])
    if True in missing_stream_ids:
        df_streams = df_streams[~missing_stream_ids]
        df_streams.reset_index()
    print("... removed {} missing stream ids".format(
        np.where(missing_stream_ids == True)[0].size))

    return(df_streams, df_churn)


def process_dataframes(df_db, df_streams, df_churn, conn):
    """
    add data to target csv
    """

    df_clean = df_churn.copy()
    df_db = df_db[np.in1d(df_db['customer_id'].values,
                          df_clean['customer_id'].values)]
    df_db.reset_index()
    unique_ids = df_clean['customer_id'].values

    # ensure we are working with correctly ordered customer_ids df_db
    if not np.array_equal(df_clean['customer_id'], df_db['customer_id']):
        raise Exception("indexes are out of order or unmatched---needs to fix")

    # query the db t create a invoice item map
    query = """
    SELECT i.invoice_item_id, i.invoice_item
    FROM INVOICE_ITEM i;
    """

    # variables for new df creation
    invoice_item_map = {d[0]: d[1] for d in conn.execute(query)}
    streams_stopped = df_streams['subscription_stopped'].values
    streams_cid = df_streams['customer_id'].values
    streams_iid = df_streams['invoice_item_id'].values
    subscriber_invoice_mode = [stats.mode(streams_iid[streams_cid == uid])[
        0][0] for uid in unique_ids]

    # create the new df
    df_clean['country'] = df_db['country']
    df_clean['gender'] = df_db['gender']
    df_clean['age'] = np.datetime64(
        'today') - df_db['DOB'].astype('datetime64')
    df_clean['age'] = [a.astype('timedelta64[Y]').astype(int)
                       for a in df_clean['age'].values]
    df_clean['customer_name'] = df_db['first_name'] + " " + df_db['last_name']
    df_clean['subscriber_type'] = [
        invoice_item_map[int(sim)] for sim in subscriber_invoice_mode]
    df_clean['num_streams'] = [
        streams_stopped[streams_cid == uid].size for uid in unique_ids]

    return(df_clean)


def update_target(target_file, df_clean, overwrite=False):
    """
    update line by line in case data are large
    """

    if overwrite or not os.path.exists(target_file):
        df_clean.to_csv(target_file, index=False)
    else:
        df_target = pd.read_csv(target_file)
        df_clean.to_csv(target_file, mode='a', index=False)


if __name__ == "__main__":

    # collect args
    arg_string = "%s -d db_filepath -s streams_filepath" % sys.argv[0]
    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'd:s:')
    except getopt.GetoptError:
        print(getopt.GetoptError)
        raise Exception(arg_string)

    # handle args
    streams_file = None
    db_file = None
    for o, a in optlist:
        if o == '-d':
            db_file = a
        if o == '-s':
            streams_file = a
    streams_file = os.path.join(DATA_DIR, streams_file)
    db_file = os.path.join(DATA_DIR, db_file)
    # print(db_file)
    target_file = os.path.join(DATA_DIR, "aavail-target.csv")

    # make the connection to the database
    conn = connect_db(db_file)

    # ingest data base data
    df_db = ingest_db_data(conn)
    df_streams, df_churn = ingest_stream_data(streams_file)
    df_clean = process_dataframes(df_db, df_streams, df_churn, conn)

    # write
    update_target(target_file, df_clean, overwrite=True)
    print("done")
