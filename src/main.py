import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import datetime
import numpy as np
from copy import deepcopy
from ohlc import plot_ohlc
from operator import itemgetter
from scipy import signal


class Window:

    def __init__(self, df, window_length, *args, **kwargs):
        '''
        :param df: DataFrame object containing the data to take a window from
        :param window_length: (int) the length of the window (in terms of rows of df)
        :param args:
        :param kwargs:
            start_from: (int) index of df, that if is given, the window will start from there
            use_df: (True|False) if set to True, window will move forward (see method: move_forward) on df till the end
                                 if set to False, new rows will have to be given to window each time "move_forward" is
                                 called.
        '''
        if 'start_from' in kwargs:
            start = kwargs['start_from']
        else:
            start = 0

        if 'use_df' in kwargs:
            self.use_df = bool(kwargs['use_df'])
            self.df = df
            self.df['Date'] = pd.to_datetime(self.df['Date'], dayfirst=True, format='%d/%m/%Y %H:%M')

        else:
            self.use_df = False

        self.w = deepcopy(df[start:start + window_length])
        self.len = window_length
        self.ext = {} # dict for extensions- for data related to the window

        if 'indicators' in kwargs:
            if 'rsi' in kwargs['indicators']:
                self.calculate_rsi()

        '''
        list of functions to call when window is changed (for example after self.move_forward
        is called)

        '''
        self.run_after_update = ['calculate_rsi', 'update_min_max']
        for i, row in self.w.iterrows():
            if isinstance(row['Date'], str):
                self.w.at[i,'Date'] = pd.to_datetime(row['Date'], dayfirst=True, format='%d/%m/%Y %H:%M')
        self.dates_list = [self.w.iloc[0]['Date'] + pd.Timedelta(x,'h') for x in range(0,self.len)]

        if self.use_df:
            self.df_index_of_last = self.w.index[-1] # note that this is the current end of window, represented by index on self.df
        self.w = self.w.reset_index(drop=True)

    def move_forward(self, n=1, row=None):
        ''' Move the window forward to next rows

        :param row: DataFrame object- row to add to the Window object, columns must match
        :param n: if use_df is True, decide how much rows to move forward on df. default is 1

        :returns: removed rows that were before in the window
        '''

        if not self.use_df and row is None:
            raise ValueError('move_forward function must get argument \'row\' if \'use_df\' is set to False')

        len_to_move = None # need for later to know the self.df_index_of_last
        if row is None:
            len_to_move = n
            # check if there is n rows to move forward to
            end_w_index = self.df_index_of_last
            end_df_index = self.df.index[-1]

            diff = end_df_index - end_w_index

            if diff < n:
                if end_w_index  == end_df_index:
                    raise ValueError('Window reached end of df, can not move forward!')
                else:
                    raise ValueError('could not move {0} steps since df has only {1} steps left.'.format(n, diff))

            to_add = self.df[end_w_index + 1: end_w_index + 1 + n]
            for i, row in to_add.iterrows():
                if isinstance(row['Date'], str):
                    to_add.at[i, 'Date'] = pd.to_datetime(row['Date'], dayfirst=True, format='%d/%m/%Y %H:%M')
            self.w = self.w.append(to_add, sort=True)

            removed = self.w.iloc[0:n]
            self.update_window(n, removed)

            self.w = self.w.iloc[n:]

        else:
            # add row to window
            row_l = len(row.index)
            len_to_move = row_l
            for i,  in row.iterrows():
                if isinstance(row.at[i,'Date'], str):
                    row.at[i,'Date'] = pd.to_datetime(row.at[i,'Date'], dayfirst=True, format='%d/%m/%Y %H:%M')
            self.w = self.w.append(row, sort=True)

            removed = self.w.iloc[0:row_l]
            self.update_window(row_l, removed)
            self.w = self.w.iloc[row_l:]

        last_removed_date = removed.iloc[-1]['Date']


        for t in ['mini', 'maxi']:
            if t in self.ext:
                for ma in self.ext[t]:
                    for date in self.ext[t][ma]['dates']:
                        if date > last_removed_date:
                            from_index = self.ext[t][ma]['dates'].index(date)
                            self.ext[t][ma]['dates'] = self.ext[t][ma]['dates'][from_index::]
                            self.ext[t][ma]['points'] = self.ext[t][ma]['points'][from_index::]
                            break

        if self.use_df:
            self.df_index_of_last = self.df_index_of_last + len_to_move
        self.w = self.w.reset_index(drop=True)



        # return removed rows
        return removed

    def update_window(self, n, removed=None):
        '''

        :param n: The number of rows added
        :return:
        '''

        self.dates_list = [self.w.iloc[0]['Date'] + pd.Timedelta(x, 'h') for x in range(0, self.len)]

        if not self.run_after_update:
            return

        if 'calculate_rsi' in self.run_after_update:
            self.calculate_rsi(n=n, removed=removed)

        if 'update_min_max' in self.run_after_update:
            self.update_min_max(n=n, removed=removed)

    def window_data(self):
        ''' Prints Data about the window. for debugging '''
        print('------------------ BRIEF ------------------')
        print('window head:')
        print(self.w.head(3))
        print('window tail:')
        print(self.w.tail(3))

        print('window length:')
        print(len(self.w.index))
        print('-------------------------------------------')
        print()

    def calculate_rsi(self, length=14, **kwargs):
        change = self.w['Close'].diff(1)
        gain = change.mask(change < 0, 0)
        loss = change.mask(change > 0, 0)

        avg_gain = gain.ewm(com=length - 1, min_periods=length).mean()
        avg_loss = loss.ewm(com=length - 1, min_periods=length).mean()

        rs = abs(avg_gain / avg_loss)

        rsi = 100 - (100 / (1 + rs))

        self.w['RSI'] = rsi

    def view(self, **kwargs):
        '''

        :param save: False to show figure, True to save it
        :param kwargs:
            'view_data': additional plotting data to add to the chart. synrax is as follow:
                         view_data = {
                            '0':{ # axe index
                                'scatter': [(x_axis,y_axis,ploting_args_dict)], # plot type
                                'plot': []
                            }
                        }
            'more_forward': (int) number of bars to plot after the end of the window (available only when 'use_df' is True)
            'more_backward': (int) number of bars to plot before the start of the window (available only when 'use_df' is True)
        '''
        window = deepcopy(self.w)

        # window.set_index('Date', inplace=True)
        start_date = window.iloc[0]['Date']
        end_date = window.iloc[-1]['Date']


        if 'more_forward' in kwargs and self.use_df:
            add = deepcopy(self.df.loc[(self.df['Date'] > self.w.iloc[-1]['Date']) & \
                                       (self.df['Date'] <= self.w.iloc[-1]['Date'] + pd.Timedelta(int(kwargs['more_forward']), 'h'))])

            if len(add.index) != kwargs['more_forward']:
                raise ValueError('Can not plot window as "more_forward" value had number greater than what df has in' +\
                 ' front of it.\n"view" got {0} to add while df had only {1}.'.format(kwargs['more_forward'], len(add.index)))

            window = window.append(add, sort=True)
            end_date += pd.Timedelta(int(kwargs['more_forward']), 'h')

        if 'more_backward' in kwargs and self.use_df:
            add = deepcopy(self.df.loc[(self.df['Date'] >= self.w.iloc[0]['Date'] - pd.Timedelta( \
                                           int(kwargs['more_backward']), 'h')) & \
                                       (self.df['Date'] < self.w.iloc[0]['Date'])])

            if len(add.index) != kwargs['more_backward']:
                raise ValueError('Can not plot window as "more_backward" value had number greater than what df has' + \
                ' behind it.\n"view" got {0} to add while df had only {1}.'.format(kwargs['more_backward'], len(add.index)))

            window = add.append(window, sort=True)
            start_date -= pd.Timedelta(int(kwargs['more_backward']), 'h')

        window.set_index('Date', inplace=True)

        # for hour timeframe
        td = end_date - start_date
        td = td.seconds / 3600 + td.days * 24



        fig, axes = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0}, sharex=True)
        axes[0], r_args = plot_ohlc(window, start_date, end_date, 'H', return_ax=True, position=[0, 0.3, 1, 0.7],
                                         use_ax=axes[0])


        date_range = [start_date + pd.Timedelta(x, 'h') for x in range(0, int(td) + 1)]

        axes[0].set_ylabel('Close')

        over_b = [70] * len(date_range)
        over_s = [30] * len(date_range)

        # axes[1].plot(date_range, window['RSI_MA_13'].values, label="RSI MA", color='b')
        axes[1].plot(date_range, window['RSI'].values, label="RSI", color='#e346cb')
        axes[1].plot(date_range, over_b, color="#63144D")
        axes[1].plot(date_range, over_s, color="#63144D")
        axes[1].set_ylabel('RSI')
        axes[1].set_xlabel('Date')

        if 'view_data' in kwargs:
            for axe in kwargs['view_data']:
                for plot_type, data in kwargs['view_data'][axe].items():
                    for data_point in data:
                        if plot_type == 'scatter':
                            axes[axe].scatter(data_point[0], data_point[1], **data_point[2])
                        if plot_type == 'plot':
                            axes[axe].plot(data_point[0], data_point[1], **data_point[2])

        axes[1].xaxis.set_major_locator(r_args['Ticks']['major_locator'])
        axes[1].xaxis.set_minor_locator(r_args['Ticks']['minor_locator'])
        axes[1].xaxis.set_major_formatter(r_args['Ticks']['major_formatter'])

        if 'title' in kwargs:
            axes[0].set_title(kwargs['title'])

        # axes[0].legend()
        # axes[1].legend()

        if 'save' in kwargs:
            # @TODO: check if file with this name already exist nd if so generate new name (for example add "(1)")
            plt.savefig(kwargs['save'], bbox_inches='tight')
            return

        plt.show()

    def get_min_max(self, col, length=1, measure='Close', **kwargs):
        '''

        :param col: str|DataFrame column - str that represent a column in the window or pass the column directly
        :param length: if col is MA or EMA, the length of the MA or EMA
        :param measure: the column to measure 'col' against
        :param savgol: amount of number to apply savgol_filter on the column before checking minimaxi points.

        :return: seperate lists of minimum and maximum points. minimum first. each list is a list of tuples in which the
                 first element if the price of the min/max point and that the date at that time (i.e. tuple(price,date)..)
        '''
        if isinstance(col, str):
            if col in self.w.columns:
                col = self.w[col]
            else:
                raise ValueError(
                    'A column named "{0}" does not exist, "get_min_max" must get valid window column name'.format(col))
        elif not isinstance(col, pd.Series):
            raise ValueError('"get_min_max" must get either column name or a pandas column object, other type was given.')

        col = col.values

        if 'savgol' in kwargs:
            #remove any nan
            nans = []
            for item in col:
                if str(item) == 'nan':
                    nans.insert(0, np.nan)
                else:
                    break

            nan_len = len(nans)
            col = col[nan_len:]
            for i in range(0,kwargs['savgol']['times']):
                col = signal.savgol_filter(col, kwargs['savgol']['window_length'], kwargs['savgol']['poly'])

            col = nans+list(col)

        diff = np.diff(col)

        trend = 0
        suspected_points = []

        # diff = [x for x in diff if str(x) != 'nan']

        for (i,), x in np.ndenumerate(diff):
            if str(x) == 'nan':
                continue
            if trend == 0:
                trend = x
                continue

            if trend * x < 0:
                suspected_points.insert(len(suspected_points), (i, trend))
                # we are taking i and not (i - 1) cause diff is 1 item less than smooth. it means that if we find suspeceted
                # point in x (which is one step further and the turning point beacuse it had created divergence) than
                # in diff the turning point in MA is (i-1) of x but in smooth is (i-1+1)=i again becacuse diff is
                # differences and each item i in diff is calculated by (i+1)-i in smooth

                trend = x

        maxi = []
        mini = []

        maxi_i = []
        mini_i = []

        deviation = range(0, -length - 1, -1)

        for point in suspected_points:
            if point[1] > 0:
                # we are looking for a maximun point
                options = []
                for p in deviation:
                    if str(self.w.iloc[point[0] + p][measure]) == 'nan':
                        continue
                    if len(mini_i) > 0 and mini_i[len(mini_i) - 1] >= point[0] + p:
                        break

                    if point[0] + p < 0 or point[0] + p > self.len - 1:
                        continue
                    options.insert(0, (point[0] + p, self.w.iloc[point[0] + p][measure]))

                if len(options) > 0:
                    max_point_index = max(options, key=itemgetter(1))[0]

                    if len(mini_i) > 0 and self.w.iloc[max_point_index][measure] <= \
                            self.w.iloc[mini_i[len(mini_i) - 1]][measure]:
                        '''
                        max point is less or equals to min point - therefore not really a max point
                        This is possible beacuse the MA is calculated by last x candles, and lets say in case
                        of threre is 1 big candle and then small ones, it may affect the MA when it stops adding it (the big
                        candle) to the equation. so the MA may temporarly do down a bit and then continue higher (and it won't be
                        a top). Example can be seen on BTCUSDT 1h chart, 14-1-2019 at 00:00, on this candle the MA 3 bottomed
                        after the previous cnadle it topped, according to this function, the prev candle it a top and the
                        current candle is a bottom, even though it is higher.
                        '''
                        # because of the above we need to delete this top and the prev bottom
                        mini_i.pop(len(mini_i) - 1)
                        mini.pop(len(mini) - 1)

                    else:
                        maxi.insert(len(maxi), tuple(
                            (self.w.iloc[max_point_index][measure], self.w.iloc[max_point_index]['Date'])))
                        maxi_i.insert(len(maxi_i), max_point_index)

            else:
                # we are looking for a minimum point
                options = []
                for p in deviation:
                    if len(maxi_i) > 0 and maxi_i[len(maxi_i) - 1] >= point[0] + p:
                        continue
                    if point[0] + p < 0 or point[0] + p > self.len - 1:
                        continue
                    options.insert(0, (point[0] + p, self.w.iloc[point[0] + p][measure]))

                if len(options) > 0:
                    min_point_index = min(options, key=itemgetter(1))[0]

                    if len(maxi_i) > 0 and self.w.iloc[min_point_index][measure] >= \
                            self.w.iloc[maxi_i[len(maxi_i) - 1]][measure]:
                        '''
                        max point is less or equals to min point - therefore not really a max point
                        This is possible beacuse the MA is calculated by last x candles, and lets say in case
                        of threre is 1 big candle and then small ones, it may affect the MA when it stops adding it (the big
                        candle) to the equation. so the MA may temporarly do down a bit and then continue higher (and it won't be
                        a top). Example can be seen it BTCUSDT 1h chart, 14-1-2019 at 00:00, on this candle the MA 3 bottomed
                        after the previous cnadle it topped, according to this function, the prev candle it a top and the
                        current candle is a bottom, even though it is higher.
                        '''
                        # because of the above we need to delete this bottom and the prev top

                        maxi.pop(len(maxi) - 1)
                        maxi_i.pop(len(maxi_i) - 1)

                    else:
                        mini.insert(len(mini), tuple(
                            (self.w.iloc[min_point_index][measure], self.w.iloc[min_point_index]['Date'])))
                        mini_i.insert(len(mini_i), min_point_index)

        return mini, maxi

    def add_min_max_based_ma(self, line, remove_if_on=None, measure='Close', **kwargs):
        ''' Automaticlly adds up min&max points to self.ext based on given MA number n

        :param line: Window column to calculate by
        :param remove_if_on: number of MA or numbers of MAs that when given, all minimaxi points in n will be deleted
        :return: None
        '''

        sp = line.split('_')
        if len(sp) == 1:
            n = 1
        else:
            n = int(sp[-1])
            if sp[0] in self.w.columns:
                self.w[line] = self.w[sp[0]].rolling(n).mean()
            else:
                raise ValueError('A column named "{0}" does not exist, "add_min_max_based_ma" ' + \
                                 'tried to create one but encountered a problem. you must provide valid window column name'.format(line))

        mini, maxi = self.get_min_max(line, n, measure, **kwargs)

        min_points, min_dates = zip(*mini)
        min_points, min_dates = list(min_points), list(min_dates)

        max_points, max_dates = zip(*maxi)
        max_points, max_dates = list(max_points), list(max_dates)

        if remove_if_on:
            if isinstance(remove_if_on, int):
                remove_if_on = [remove_if_on]

            for ma in remove_if_on:
                for date in self.ext['maxi'][ma]['dates']:
                    if date in max_dates:
                        i = max_dates.index(date)
                        max_dates.pop(i)
                        max_points.pop(i)

                for date in self.ext['mini'][ma]['dates']:
                    if date in min_dates:
                        i = min_dates.index(date)
                        min_dates.pop(i)
                        min_points.pop(i)

        if 'mini' in self.ext:
            self.ext['mini'][n]={
                'dates': min_dates,
                'points': min_points,
                'measure': measure
            }
        else:
            self.ext['mini'] = {
                n: {
                    'dates': min_dates,
                    'points': min_points,
                    'measure': measure
                }
            }

        if 'maxi' in self.ext:
            self.ext['maxi'][n] = {
                'dates': max_dates,
                'points': max_points,
                'measure': measure
            }
        else:
            self.ext['maxi'] = {
                n: {
                    'dates': max_dates,
                    'points': max_points,
                    'measure': measure
                }
            }

    def update_min_max(self, n, **kwargs):
        ''' More conveniently define minimaxi points

        :param n: the number of the MA
        :param kwargs:
        :return: None
        '''
        MAs = list(self.ext['mini'].keys())

        for ma in MAs:
            measure = self.ext['mini'][ma]['measure']
            if ma == 1:
                ma_name = measure
            else:
                ma_name = measure + '_' + str(ma)
            self.w[ma_name] = self.w['RSI'].rolling(ma).mean()


            diff = np.diff(self.w.iloc[-n-2::][ma_name])
            trend = 0
            deviation = range(0, -n - 1, -1)

            for (i, d) in enumerate(diff):
                if i+1 == len(diff):
                    break
                if trend == 0:
                    trend = diff[i]
                if trend * diff[i+1] < 0: # a turning point
                    if trend > 0: # it's a max point
                        max_point = None
                        for p in deviation:
                            if max_point is None:
                                max_point = [self.w.iloc[(i-n-1) + p][measure],(i-n-1) + p]
                                continue
                            if self.w.iloc[(i-n-1) + p]['Date'] == self.ext['mini'][ma]['dates'][-1]:
                                # we reached the previous minimum point- stop the loop
                                break

                            if len(self.ext['mini'][ma]['dates']) > 0 and self.w.iloc[(i-n-1) + p][measure] <= \
                                    self.ext['mini'][ma]['points'][-1]:
                                self.ext['mini'][ma]['points'].pop(-1)
                                self.ext['mini'][ma]['dates'].pop(-1)
                                max_point = None
                                break
                            if max_point[0] < self.w.iloc[(i-n-1) + p][measure]:
                                max_point = [self.w.iloc[(i-n-1) + p][measure],(i-n-1) + p]

                        if max_point is None:
                            continue

                        self.ext['maxi'][ma]['dates'].insert(len(self.ext['maxi'][ma]['dates']), \
                                                             self.w.iloc[max_point[1]]['Date'])
                        self.ext['maxi'][ma]['points'].insert(len(self.ext['maxi'][ma]['points']), \
                                                             max_point[0])

                    elif trend < 0:  # it's a min point
                        min_point = None
                        for p in deviation:
                            if min_point is None:
                                min_point = [self.w.iloc[(i - n - 1) + p][measure], (i - n - 1) + p]
                                continue
                            if self.w.iloc[(i - n - 1) + p]['Date'] == self.ext['maxi'][ma]['dates'][-1]:
                                # we reached the previous maximum point- stop the loop
                                break

                            if len(self.ext['maxi'][ma]['dates']) > 0 and self.w.iloc[(i - n - 1) + p][measure] >= \
                                    self.ext['maxi'][ma]['points'][-1]:
                                self.ext['maxi'][ma]['points'].pop(-1)
                                self.ext['maxi'][ma]['dates'].pop(-1)
                                min_point = None
                                break
                            if min_point[0] > self.w.iloc[(i - n - 1) + p][measure]:
                                min_point = [self.w.iloc[(i - n - 1) + p][measure], (i - n - 1) + p]

                        if min_point is None:
                            continue

                        self.ext['mini'][ma]['dates'].insert(len(self.ext['mini'][ma]['dates']), \
                                                             self.w.iloc[min_point[1]]['Date'])
                        self.ext['mini'][ma]['points'].insert(len(self.ext['mini'][ma]['points']), \
                                                              min_point[0])

                    trend = diff[i+1]

    def find_rsi_divergence(self, macro=None, micro=None):

        if macro is None:
            macro = max(list(self.ext['mini'].keys()))
        if micro is None:
            micro = min(list(self.ext['mini'].keys()))

        current_rsi_value = self.w.iloc[-1]['RSI']

        if current_rsi_value > 50:
            dates = self.ext['maxi'][micro]['dates']
            points = self.ext['maxi'][micro]['points']
            for i in range(len(dates)-1, -1, -1):
                if dates[i] < self.ext['maxi'][macro]['dates'][-2]:
                    break


                win_index = self.w.index[self.w['Date'] == dates[i]].tolist()[0]


                if points[i] > current_rsi_value and self.w.iloc[win_index]['Close'] < self.w.iloc[-1]['Close'] and points[i] >= 70:
                    if len(points) == i+1:
                        return (win_index, self.w.index[-1] - self.w.index[0])

                    # before deciding let's see there isn't some point higher between those two:
                    max_rsi_between = max(points[i+1::])
                    if max_rsi_between >= points[i]:
                        # no real divergence
                        break

                    max_close_between = max(self.w.loc[(self.w.index > win_index) & \
                                                       (self.w.index < self.w.index[-1])]['Close'].values)
                    if max_close_between >= self.w.iloc[-1]['Close']:
                        # no real divergence
                        break

                    # if we made it to hear than this is a real divergence
                    return  (win_index, self.w.index[-1] - self.w.index[0])

        else:
            dates = self.ext['mini'][micro]['dates']
            points = self.ext['mini'][micro]['points']
            for i in range(len(dates) - 1, -1, -1):
                if dates[i] < self.ext['mini'][macro]['dates'][-2]:
                    break

                win_index = self.w.index[self.w['Date'] == dates[i]].tolist()[0]

                if points[i] < current_rsi_value and self.w.iloc[win_index]['Close'] > self.w.iloc[-1]['Close'] and points[i] <= 30:
                    if len(points) == i+1:
                        return (win_index, self.w.index[-1] - self.w.index[0])

                    # before deciding let's see there isn't some point lower between those two:
                    min_rsi_between = min(points[i + 1::])
                    if min_rsi_between <= points[i]:
                        # no real divergence
                        break

                    min_close_between = min(self.w.loc[(self.w.index > win_index) & \
                                                       (self.w.index < self.w.index[-1])]['Close'].values)
                    if min_close_between <= self.w.iloc[-1]['Close']:
                        # no real divergence
                        break

                    # if we made it to hear than this is a real divergence
                    return (win_index, self.w.index[-1] - self.w.index[0])

        return False
        
        
        
