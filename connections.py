from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Numeric
from sqlalchemy.orm import sessionmaker, relationship, backref, aliased, mapper
import sqlalchemy as al
import datetime as dt
from sqlalchemy import (MetaData, Table, Column, Integer,
                        Date, select, literal, and_, or_, exists, func)
from sqlalchemy.inspection import inspect
from sqlalchemy.dialects.mysql import BIGINT
from sqlalchemy.exc import DisconnectionError
import sqlalchemy.pool as pool


connectionString = 'mysql://USER:%s@127.0.0.1/TABLE_NAME?charset=utf8mb4' % urlquote('PASSWORD')

def returnEngine():
    connection_string = connectionString

    db_engine = create_engine(connection_string,
                            pool_recycle = 3600, pool_size=100, pool_pre_ping=True, encoding='UTF8') #
    event.listen(db_engine, 'checkout', checkout_listener)
    return db_engine


def checkout_listener(dbapi_con, con_record, con_proxy):
    try:
        try:
            dbapi_con.ping(False)
        except TypeError:
            dbapi_con.ping()
    except dbapi_con.OperationalError as exc:
        if exc.args[0] in (2006, 2013, 2014, 2045, 2055):
            raise DisconnectionError()
        else:
            raise


def initializeSession():
    print('Initializing session with initializeSession()')
    engine = returnEngine()
    conn = engine.connect()
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    metadata = al.schema.MetaData(bind=engine, reflect=True)
    return session, metadata, conn, engine
	
session, metadata, conn, engine = initializeSession()


#-----------------------------------

conn.execute("INSERT bla bla")
