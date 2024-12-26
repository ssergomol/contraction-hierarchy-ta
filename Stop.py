import json
import csv

class Stop:
    def __init__(self, StopId, Code, Name, StopType, Zone, Ward,
                 AddressNo, Street, SupportDisability, Status, Lng, Lat, Search, Routes, RouteId, RouteVarId):
        self.StopId = StopId
        self.Code = Code
        self.Name = Name
        self.StopType = StopType
        self.Zone = Zone
        self.Ward = Ward
        self.AddressNo = AddressNo
        self.Street = Street
        self.SupportDisability = SupportDisability
        self.Status = Status
        self.Lng = Lng
        self.Lat = Lat
        self.Search = Search
        self.Routes = Routes
        self.RouteId = RouteId
        self.RouteVarId = RouteVarId

    def set_stop_id(self, stop_id):
        self.StopId = stop_id

    def set_code(self, code):
        self.Code = code

    def set_name(self, name):
        self.Name = name

    def set_stop_type(self, stop_type):
        self.StopType = stop_type

    def set_zone(self, zone):
        self.Zone = zone

    def set_ward(self, ward):
        self.Ward = ward

    def set_address_no(self, address_no):
        self.AddressNo = address_no

    def set_street(self, street):
        self.Street = street

    def set_support_disability(self, support_disability):
        self.SupportDisability = support_disability

    def set_status(self, status):
        self.Status = status

    def set_lng(self, lng):
        self.Lng = lng

    def set_lat(self, lat):
        self.Lat = lat

    def set_search(self, search):
        self.Search = search

    def set_routes(self, routes):
        self.Routes = routes

    def set_route_id(self, route_id):
        self.RouteId = route_id

    def set_route_var_id(self, route_var_id):
        self.RouteVarId = route_var_id

    def get_stop_id(self):
        return self.StopId

    def get_code(self):
        return self.Code

    def get_name(self):
        return self.Name

    def get_stop_type(self):
        return self.StopType

    def get_zone(self):
        return self.Zone

    def get_ward(self):
        return self.Ward

    def get_address_no(self):
        return self.AddressNo

    def get_street(self):
        return self.Street

    def get_support_disability(self):
        return self.SupportDisability

    def get_status(self):
        return self.Status

    def get_lng(self):
        return self.Lng

    def get_lat(self):
        return self.Lat

    def get_search(self):
        return self.Search

    def get_routes(self):
        return self.Routes

    def get_route_id(self):
        return self.RouteId

    def get_route_var_id(self):
        return self.RouteVarId

    def get_stop(self):
        return {
            'StopId': self.StopId,
            'Code': self.Code,
            'Name': self.Name,
            'StopType': self.StopType,
            'Zone': self.Zone,
            'Ward': self.Ward,
            'AddressNo': self.AddressNo,
            'Street': self.Street,
            'SupportDisability': self.SupportDisability,
            'Status': self.Status,
            'Lng': self.Lng,
            'Lat': self.Lat,
            'Search': self.Search,
            'Routes': self.Routes,
            'RouteId': self.RouteId,
            'RouteVarId': self.RouteVarId
        }

class StopQuery:
    def __init__(self):
        self.stops = []

    def read_stop(self):
        try:
            with open('input_data/stops.json', 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    for stop in data['Stops']:
                        stop['RouteId'] = data['RouteId']
                        stop['RouteVarId'] =  data['RouteVarId']
                        stopObject = Stop(**stop)
                        self.stops.append(stopObject)
        except Exception as a:
            print(f"Error: {a}")

    def searchByABC(self, **kwargs):
        fieldnames = ['StopId', 'Code', 'Name', 'StopType', 'Zone', 'Ward',
                      'AddressNo', 'Street', 'SupportDisability', 'Status', 'Lng', 'Lat', 'Search', 'Routes', 'RouteId', 'RouteVarId']
        result = []
        for stop in self.stops:
            check = True
            for key in kwargs:
                attribute = getattr(stop, key, 'Unknown')
                query_value = kwargs[key]
                if attribute == 'Unknown':
                    print(f'{key} attribute is not found')
                    print(f'Please have a query with those attribute {fieldnames}')
                    return
                if attribute != query_value:
                    check = False
                    break
            if check:
                result.append(stop)
        if len(result) == 0:
            print(f"No stops found!")
        return result

    def outputAsCSV(self, list):
        fieldnames = ['StopId', 'Code', 'Name', 'StopType', 'Zone', 'Ward',
                         'AddressNo', 'Street', 'SupportDisability', 'Status', 'Lng', 'Lat', 'Search', 'Routes', 'RouteId', 'RouteVarId']
        with open('output/stop.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            list = [element.get_stop() for element in list]
            writer.writerows(list)

    def outputAsJson(self, list):
        with open('output/stop.json', 'w', newline='', encoding='utf-8') as f:
            list = [element.get_stop() for element in list]
            for data in list:
                json_object = json.dumps(data, ensure_ascii=False)
                f.write(json_object + '\n')
