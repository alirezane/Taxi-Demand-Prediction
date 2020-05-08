import pandas as pd
import numpy as np
import datetime


def clean_date(date):
    date = str(date)
    return date[:19]


def round_down(num, divisor):
    return num - (num % divisor)


def unique_rides(df, interval):
    df['date'] = df.createdAt.apply(datetime.datetime.date)
    df['hour'] = df.createdAt.apply(lambda x: x.hour)
    df['minute'] = df.createdAt.apply(lambda x: round_down(x.minute, interval))
    df = df.drop_duplicates(subset=['passengerId', 'date', 'hour', 'minute'], keep="first")
    df = df.drop(['date', 'hour', 'minute'], axis=1)
    return df


def add_zeros(df, max_date, min_date, interval):
    start = str(min_date) + " 00:00:00"
    end = str(max_date) + " 23:55:00"
    dt = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    end = datetime.datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
    step = datetime.timedelta(minutes=interval)
    result = []
    while dt < end:
        result.append(dt)
        dt += step
    date_truncs = result
    date_truncs = pd.DataFrame(date_truncs, columns=['createdAt'])
    date_truncs['region']=df.region.max()
    df = pd.merge(date_truncs, df, how='left', on=['createdAt', 'region'])
    return df


def count_rides(df, max_date, min_date, time_interval):
    df['createdAt'] = df.createdAt.apply(lambda x: x.replace(second=0, minute=round_down(x.minute, time_interval)))
    df['travel_time_rate'] = df.estimatedETA / df.googleDistance
    df.travel_time_rate.fillna(0, inplace=True)
    df = df.groupby(['region', 'createdAt'], as_index=False)\
        .agg({'id': {"ride_count": ['count']}, 'travel_time_rate': {"travel_time_rate": ['mean']}})
    df.columns = df.columns.droplevel(1)
    df = add_zeros(df, max_date, min_date, time_interval)
    df.ride_count.fillna(0, inplace=True)
    df.travel_time_rate = df.travel_time_rate.replace(0, np.nan)
    return df


class DataPrepare:
    def __init__(self, grid_dim, aggregation_interval, unique_interval=20):
        self.aggregation_interval = aggregation_interval
        self.unique_interval = unique_interval
        self.grid_dim = grid_dim
        self.read_data()
        self.eliminate_noisy_data()
        self.clean_dates()
        self.make_grids()
        self.unique_rides()
        self.aggregate()
        self.export_data()

    def read_data(self):
        self.ridereq_data = pd.read_csv('./data/Shomara.csv',
                                        names=['id', 'createdAt', 'googleDistance', 'estimatedETA',
                                               'passengerId', 'status', 'neighbourhoodCode',
                                               'originLat', 'originLong', 'cancellationTime'])

    def eliminate_noisy_data(self):
        self.ridereq_data = self.ridereq_data[(self.ridereq_data.estimatedETA > 180) &
                                              (self.ridereq_data.estimatedETA < 108000) &
                                              (self.ridereq_data.googleDistance > 0) &
                                              (self.ridereq_data.googleDistance < 65000) &
                                              (self.ridereq_data.googleDistance /
                                               self.ridereq_data.estimatedETA < 28)]

    def clean_dates(self):
        self.ridereq_data['createdAt'] = self.ridereq_data.createdAt.apply(clean_date)
        self.ridereq_data['createdAt'] = self.ridereq_data.createdAt.apply(str)
        self.ridereq_data['createdAt'] = pd.to_datetime(self.ridereq_data['createdAt'],
                                                        format='%Y-%m-%d %H:%M:%S')
        self.min_date = datetime.datetime.date(min(self.ridereq_data['createdAt']))
        self.max_date = datetime.datetime.date(max(self.ridereq_data['createdAt']))

        self.ridereq_data['cancellationTime'] = self.ridereq_data[self.ridereq_data.status == 'CANCELED']\
            .cancellationTime.apply(clean_date)
        self.ridereq_data['cancellationTime'] = pd.to_datetime(self.ridereq_data['cancellationTime'],
                                                               format='%Y-%m-%d %H:%M:%S')
        self.ridereq_data['cancellationTime'] = self.ridereq_data['cancellationTime'] - \
                                                self.ridereq_data['createdAt']
        self.ridereq_data = self.ridereq_data[(self.ridereq_data.status != 'CANCELED') |
                                              (self.ridereq_data.cancellationTime >
                                               datetime.timedelta(seconds=10))]

    def make_grids(self):
        self.ridereq_data['lat_code'] = self.ridereq_data.originLat\
            .apply(lambda x: str(int(x / ((35.831777 - 35.567994) / 16))))
        self.ridereq_data['long_code'] = self.ridereq_data.originLong\
            .apply(lambda x: str(int(x / ((51.606610 - 51.111395) / 16))))
        self.ridereq_data['region'] = self.ridereq_data['lat_code'] + self.ridereq_data['long_code']

        self.ridereq_data = self.ridereq_data.sort_values(['region', 'createdAt'], ascending=[1, 1])
        self.ridereq_data = self.ridereq_data.groupby('region', as_index=False)

    def unique_rides(self):
        self.ridereq_data = self.ridereq_data.apply(lambda x: unique_rides(x, self.unique_interval))
        self.ridereq_data = self.ridereq_data.groupby('region', as_index=False)

    def aggregate(self):
        self.ridereq_data = self.ridereq_data.apply(lambda x: count_rides(x, self.max_date, self.min_date,
                                                                          self.aggregation_interval))

    def export_data(self):
        name = "./Data/ridereqs-" + str(self.grid_dim) + "X" + str(self.grid_dim) + "-" +\
               str(self.aggregation_interval) + "min.csv"
        self.ridereq_data.to_csv(name, index=False)


DataPrepare(16, 5)

