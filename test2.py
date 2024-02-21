from cgi import parse_multipart
from datetime import date
import os.path
from unicodedata import UCD
import pandas as pd
import logging
import numpy as np

today = date.today()
d = today.strftime("%d_%B_%Y")
year = today.strftime("%Y")
logging.info(f'starting {d}')
## needs to create empty file first with cartridge numbers for this script to run
filepath = f'/storage1/fs1/rvmartin/Active/SPARTAN-shared/Analysis_Data/FTIR/UCDavis_samples_log/{year}/{d}_empty.xlsx'

# if os.path.exists(filepath):
Davissheet = pd.ExcelFile(filepath)


def getCartridgeNum():
    UCsheet = pd.read_excel(filepath)
    UCDavis = UCsheet['Cartridge Number']
    UCDavis = UCDavis.repeat(8)
    return UCDavis


UCDavis = getCartridgeNum()
UCDavis.index = range(0, len(UCDavis))


def getSitesCode():
    SitesCode = UCDavis.str[:4]  # a series to store location codes
    return SitesCode


SitesCode = getSitesCode()


def getpartMTL():
    i = 0
    Filterset_Offset = 8
    partMTL = pd.DataFrame()
    while (i < len(SitesCode)):
        if 'Lab' in SitesCode[i]:
            partMTL.loc[i, :] = 'NaN'  # add a row of NaN
            i = i + 1  # need to leave out the space for NaN or empty
        else:
            MTL_masses_path = f'/storage1/fs1/rvmartin/Active/SPARTAN-shared/Analysis_Data/Filter_Masses/Masses_by_site/MTL_weighing_WashU/{SitesCode[i]}_MTL_masses.csv'
            MTL_masses = pd.read_csv(MTL_masses_path)  # a dataframe
            partMTL = pd.concat([partMTL, MTL_masses[MTL_masses['CartridgeID'] == UCDavis[
                i]]])  # all columns corresponding to Cartridge number given, dataframe
            # partMTL=pd.concat([partMTL,MTL_masses[MTL_masses['CartridgeID']==CartridgeID[i]]]) #all columns corresponding to Cartridge number given, dataframe
            i = i + Filterset_Offset
    partMTL.index = range(0, len(partMTL))
    return partMTL


def getProjectID():
    SMlist = ['ETAD', 'ILHA', 'ILNZ', 'INDH', 'TWKA', 'TWTA', 'USPA', 'ZAJB', 'ZAPR', 'CHTS']
    ProjectID = pd.Series()  # a series to record ProjectID
    SM = pd.Series(['SM'])
    SS = pd.Series(['SS'])
    i = 0
    while (i < len(SitesCode)):
        if SitesCode[i] in SMlist:
            ProjectID = pd.concat([ProjectID, SM])
        else:
            ProjectID = pd.concat([ProjectID, SS])
        i = i + 1
    ProjectID.index = range(0, len(ProjectID))
    return ProjectID


def getFilterType():
    SMFT = pd.Series(['PM2.5', 'PM2.5', 'PM2.5', 'PM2.5', 'PM2.5', 'PM2.5', 'FB', 'PM2.5'])
    SSFT = pd.Series(['PM2.5', 'PM2.5', 'PM2.5', 'PM2.5', 'PM2.5', 'PM2.5', 'FB', 'PM10'])
    LabFT = pd.Series(['LB'])
    FilterType = pd.Series()  # a series to record FilterType
    i = 0
    Filterset_Offset = 8
    while (i < len(SitesCode)):
        if ProjectID[i] == 'SM':
            FilterType = pd.concat([FilterType, SMFT])
            i = i + Filterset_Offset
        elif 'Lab' in SitesCode[i]:
            FilterType = pd.concat([FilterType, LabFT])
            i = i + 1
        else:
            FilterType = pd.concat([FilterType, SSFT])
            i = i + Filterset_Offset
    FilterType.index = range(0, len(FilterType))
    return FilterType


def getpart_dates_flows():
    i = 0
    part_dates_flows = pd.DataFrame()
    while (i < len(SitesCode)):
        if 'Lab' in SitesCode[i]:
            part_dates_flows.loc[i, :] = 0  # add a row of 0
            i = i + 1
        else:
            dates_flows_path = f'/storage1/fs1/rvmartin/Active/SPARTAN-shared/Site_Sampling/symlinks_for_automation/{SitesCode[i]}/{SitesCode[i]}_dates_flows.xlsx'
            dates_flows = pd.read_excel(dates_flows_path)  # a dataframe
            part_dates_flows = pd.concat([part_dates_flows, dates_flows[
                dates_flows['Analysis_ID'] == partMTL['AnalysisID'][
                    i]]])  # all columns corresponding to Analysis ID given, dataframe
            i = i + 1
    part_dates_flows.index = range(0, len(part_dates_flows))
    return part_dates_flows


if __name__ == "__main__":
    partMTL = getpartMTL()
    FilterID = partMTL['FilterID'].rename("Filter ID")  # a series to record FilterID
    AnalysisID = partMTL['AnalysisID'].rename("Analysis ID")  # a series to record AnalysisID
    Barcode = partMTL['Filter_Barcode'].rename("Barcode")  # a series to record Barcode
    massCollect = partMTL['Net_Weight_ug'].rename(
        "Mass collected on filter (ug)")  # a series to record mass collected on filter
    ProjectID = getProjectID().rename("Project ID")  # a series to record ProjectID
    FilterType = getFilterType().rename("Filter Type")  # a series to record FilterType
    part_dates_flows = getpart_dates_flows()
    sampleVolume = part_dates_flows['volume_m3'].rename("Sampled volume (m3)")  # a series of sampled volumes

    part_dates_flows['start_month'] = part_dates_flows['start_month'].astype(str)
    part_dates_flows['start_month'] = part_dates_flows['start_month'].str.replace('.0', '', regex=False)
    part_dates_flows['start_day'] = part_dates_flows['start_day'].astype(str)
    part_dates_flows['start_day'] = part_dates_flows['start_day'].str.replace('.0', '', regex=False)
    part_dates_flows['start_year'] = part_dates_flows['start_year'].astype(str)
    part_dates_flows['start_year'] = part_dates_flows['start_year'].str.replace('.0', '', regex=False)
    part_dates_flows['stop_month'] = part_dates_flows['stop_month'].astype(str)
    part_dates_flows['stop_month'] = part_dates_flows['stop_month'].str.replace('.0', '', regex=False)
    part_dates_flows['stop_day'] = part_dates_flows['stop_day'].astype(str)
    part_dates_flows['stop_day'] = part_dates_flows['stop_day'].str.replace('.0', '', regex=False)
    part_dates_flows['stop_year'] = part_dates_flows['stop_year'].astype(str)
    part_dates_flows['stop_year'] = part_dates_flows['stop_year'].str.replace('.0', '', regex=False)
    startDate = part_dates_flows['start_month'] + "/" + part_dates_flows['start_day'] + "/" + part_dates_flows[
        'start_year']
    startDate = startDate.rename("Sampling Start Date")  # a series of start dates
    endDate = part_dates_flows['stop_month'] + "/" + part_dates_flows['stop_day'] + "/" + part_dates_flows['stop_year']
    endDate = endDate.rename("Sampling End Date")  # a series of end dates
    # format startDate and endDate columns
    i = 0
    while (i < len(startDate)):
        if startDate[i] == '0/0/0':
            startDate[i] = '0'
        elif startDate[i] == 'nan/nan/nan':
            startDate[i] = 'NaN'
        if endDate[i] == '0/0/0':
            endDate[i] = '0'
        elif endDate[i] == 'nan/nan/nan':
            endDate[i] = 'NaN'
        i = i + 1
    massCollect = massCollect.astype(float)
    UCDavis = pd.concat(
        [UCDavis, Barcode, FilterID, AnalysisID, FilterType, ProjectID, startDate, endDate, massCollect, sampleVolume],
        axis=1)
    UCDavis["Shipment ID (Date)"] = ""
    UCDavis["Lot ID"] = ""
    UCDavis["Comments"] = ""
    UCDavis = UCDavis[
        ['Shipment ID (Date)', 'Cartridge Number', 'Barcode', 'Filter ID', 'Analysis ID', 'Filter Type', 'Project ID',
         'Lot ID', 'Sampling Start Date', 'Sampling End Date', 'Mass collected on filter (ug)', 'Sampled volume (m3)',
         'Comments']]
    UCDavis.to_excel(
        f'/storage1/fs1/rvmartin/Active/SPARTAN-shared/Analysis_Data/FTIR/UCDavis_samples_log/{year}/next_UCDavis_shipment_{d}.xlsx',
        index=False, na_rep='NaN')


