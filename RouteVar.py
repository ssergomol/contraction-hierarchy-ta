import csv
import json
class RouteVar:
    def __init__(self, RouteId, RouteVarId, RouteVarName, RouteVarShortName,
                 RouteNo, StartStop, EndStop, Distance, Outbound, RunningTime):
        self.RouteId = RouteId
        self.RouteVarId = RouteVarId
        self.RouteVarName = RouteVarName
        self.RouteVarShortName = RouteVarShortName
        self.RouteNo = RouteNo
        self.StartStop = StartStop
        self.EndStop = EndStop
        self.Distance = Distance
        self.Outbound = Outbound
        self.RunningTime = RunningTime

    def set_route_id(self, RouteId):
        self.RouteId = RouteId

    def set_route_var_id(self, RouteVarId):
        self.RouteVarId = RouteVarId

    def set_route_var_name(self, RouteVarName):
        self.RouteVarName = RouteVarName

    def set_route_var_short_name(self, RouteVarShortName):
        self.RouteVarShortName = RouteVarShortName

    def set_route_no(self, RouteNo):
        self.RouteNo = RouteNo

    def set_start_stop(self, StartStop):
        self.StartStop = StartStop

    def set_end_stop(self, EndStop):
        self.EndStop = EndStop

    def set_distance(self, Distance):
        self.Distance = Distance

    def set_outbound(self, Outbound):
        self.Outbound = Outbound

    def set_running_time(self, RunningTime):
        self.RunningTime = RunningTime

    def get_route_id(self):
        return self.RouteId

    def get_route_var_id(self):
        return self.RouteVarId

    def get_route_var_name(self):
        return self.RouteVarName

    def get_route_var_short_name(self):
        return self.RouteVarShortName

    def get_route_no(self):
        return self.RouteNo

    def get_start_stop(self):
        return self.StartStop

    def get_end_stop(self):
        return self.EndStop

    def get_distance(self):
        return self.Distance

    def get_outbound(self):
        return self.Outbound

    def get_running_time(self):
        return self.RunningTime

        # Getter methods
    def get_route_var(self):
        return {
            'RouteId': self.RouteId,
            'RouteVarId': self.RouteVarId,
            'RouteVarName': self.RouteVarName,
            'RouteVarShortName': self.RouteVarShortName,
            'RouteNo': self.RouteNo,
            'StartStop': self.StartStop,
            'EndStop': self.EndStop,
            'Distance': self.Distance,
            'Outbound': self.Outbound,
            'RunningTime': self.RunningTime

        }


class RouteVarQuery:
    def __init__(self):
        self.route_vars = []

    def readRouteVar(self):
        try:
            with open('input_data/vars.json', 'r', encoding='utf-8') as f:
                for line in f:
                    d = json.loads(line)
                    for var in d:
                        route_var = RouteVar(**var)
                        self.route_vars.append(route_var)
        except Exception as a:
            print(f"Error: {a}")

    def searchByABC(self, **kwargs):
        fieldnames = ['RouteId', 'RouteVarId', 'RouteVarName', 'RouteVarShortName', 'RouteNo', 'StartStop', 'EndStop',
                      'Distance', 'Outbound', 'RunningTime']
        result = []
        for route_var in self.route_vars:
            check = True
            for key in kwargs:
                attribute = getattr(route_var, key, 'Unknown')
                query_value = kwargs[key]
                if attribute == 'Unknown':
                    print(f'{key} attribute is not found')
                    print(f'Please have a query with those attribute {fieldnames}')
                    return
                if attribute != query_value:
                    check = False
                    break
            if check:
                result.append(route_var)
        if len(result) == 0:
            print(f"No routes found!")
        return result

    def outputAsCSV(self, list):
        fieldnames = ['RouteId', 'RouteVarId', 'RouteVarName', 'RouteVarShortName', 'RouteNo', 'StartStop', 'EndStop',
                      'Distance', 'Outbound', 'RunningTime']
        with open('output/route_var.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            list = [element.get_route_var() for element in list]
            writer.writerows(list)

    def outputAsJson(self, list):
        with open('output/route_var.json', 'w', newline='', encoding='utf-8') as f:
            list = [element.get_route_var() for element in list]
            for data in list:
                json_object = json.dumps(data, ensure_ascii=False)
                f.write(json_object + '\n')
