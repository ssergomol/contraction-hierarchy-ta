import json
import itertools
import csv

class Path:
    def __init__(self, lat, lng, RouteId, RouteVarId):
        self.lat = lat
        self.lng = lng
        self.RouteId = RouteId
        self.RouteVarId = RouteVarId

    def set_lat(self, lat):
        self.lat = lat

    def set_lng(self, lng):
        self.lng = lng

    def set_route_id(self, RouteId):
        self.RouteId = RouteId

    def set_route_var_id(self, RouteVarId):
        self.RouteVarId = RouteVarId

    def get_lat(self):
        return self.lat

    def get_lng(self):
        return self.lng

    def get_route_id(self):
        return self.RouteId

    def get_route_var_id(self):
        return self.RouteVarId

    def get_path(self):
        return {
            'lat': self.lat,
            'lng': self.lng,
            'RouteId': self.RouteId,
            'RouteVarId': self.RouteVarId
        }

class PathQuery:
    def __init__(self):
        self.paths = []

    def read_path(self):
        try:
            with open('input_data/paths.json', 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    for (a, b) in zip(data['lat'], data['lng']):
                        #print(a, b, data['RouteId'], data['RouteVarId'])
                        path = Path(a, b, data['RouteId'], data['RouteVarId'])
                        self.paths.append(path)
        except Exception as a:
            print(f"Error: {a}")

    def searchByABC(self, **kwargs):
        fieldnames = ['lat', 'lng', 'RouteId', 'RouteVarId']
        result = []
        for path in self.paths:
            check = True
            for key in kwargs:
                attribute = getattr(path, key, 'Unknown')
                query_value = kwargs[key]
                if attribute == 'Unknown':
                    print(f'{key} attribute is not found')
                    print(f'Please have a query with those attribute {fieldnames}')
                    return
                if str(attribute) != str(query_value):
                    check = False
                    break
            if check:
                result.append(path)
        if len(result) == 0:
            print(f"No paths found!")
        return result

    def outputAsCSV(self, list):
        fieldnames = ['lat', 'lng', 'RouteId', 'RouteVarId']
        with open('output/pathOutput.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            list = [element.get_path() for element in list]
            writer.writerows(list)

    def outputAsJson(self, list):
        with open('output/pathOutput.json', 'w', newline='', encoding='utf-8') as f:
            list = [element.get_path() for element in list]
            for data in list:
                json_object = json.dumps(data, ensure_ascii=False)
                f.write(json_object + '\n')
