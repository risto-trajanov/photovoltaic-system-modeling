from datetime import date, timedelta, datetime

import numpy
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from utils import persist_data
from defusedxml import lxml


def save_response(resp):
    persist_data('tsdb_sma_energy', resp)


class SMA:
    # URLs usefull for navigating through sunnyportal
    BASE_URL = 'https://www.sunnyportal.com'
    LOGIN_URL = '/Templates/Start.aspx'
    OPEN_INVERTER_URL = '/FixedPages/InverterSelection.aspx'
    SET_FILE_DATE_URL = '/FixedPages/AnalysisTool.aspx'
    CURRENT_PRODUCTION_URL = '/Dashboard?_=1'
    DOWNLOAD_RESULTS_URL = '/Templates/DownloadDiagram.aspx'
    DASHBOARD_URL = '/FixedPages/Dashboard.aspx'
    EXAMPLE_PLANTS = '/Templates/ExamplePlants.aspx'

    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.session = requests.Session()

    # We need to go through a sequence of pages so we can get to the page we desiere
    # Problem occured: We could not get directly to the api page so we can extract our data
    # we went through series of pages so we kind of simulated page interaction so we can get to
    # our PV system
    def login(self, plantOid):
        response = self.session.get(SMA.BASE_URL + SMA.LOGIN_URL)
        self.session.get(SMA.BASE_URL + SMA.EXAMPLE_PLANTS)
        response = self.session.get(SMA.BASE_URL + '/RedirectToPlant/' + plantOid)
        self.plantOid = plantOid
        return response

    def request(self, date):
        params = {
            'ID': 'e175e9da-739c-47f0-ba9e-456aad2d0887',
            'endTime': date.strftime("%m/%d/%Y %I:%M:%S %p"),
            'splang': 'en-US', 'plantTimezoneBias': '180', 'name': ''
        }
        response = self.session.get('https://www.sunnyportal.com/Templates/PublicChartValues.aspx', params=params)

        return response.text

    def request_and_parse(self, date):
        return self.parse_response(self.request(date), date)

    def parse_response(self, resp, date):
        data_global = pd.DataFrame()
        html = resp
        soup = BeautifulSoup(html, 'lxml')
        my_element_id = 'ctl00_ContentPlaceHolder1_UserControlChartValues1_Table1'
        content = soup.find('table', id=my_element_id)
        table_body = content
        data = []
        rows = table_body.find_all('tr')

        df_rows = list()
        for row in rows:
            cols = row.find_all("td")
            hour = cols[0].text
            power = cols[1].text

            time_str = date.strftime('%m/%d/%Y')
            time_str = time_str + ' ' + hour
            try:
                datetime_object = datetime.strptime(time_str, '%m/%d/%Y %I:%M %p')
                power = float(power)
            except:
                continue

            row = pd.Series([datetime_object, power])
            df_rows.append(pd.DataFrame([row]))

        energy_data = pd.concat(df_rows). \
            replace([numpy.nan], 0). \
            rename(columns={0: 'datetime', 1: 'energy'}). \
            set_index('datetime')

        return energy_data


def main():
    sma_requestor = SMA('', '')
    sma_requestor.login('6f2ec9b6-063e-4945-afb2-e6ee7c45f615')
    for i in range(1, 31):
        resp = sma_requestor.request_and_parse(datetime(2020, 6, i, 23, 59, 59))
        assert isinstance(resp, pd.DataFrame)
        assert isinstance(resp.index, pd.DatetimeIndex)
        save_response(resp)


if __name__ == '__main__':
    main()
