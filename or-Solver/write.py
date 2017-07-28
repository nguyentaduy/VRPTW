towrite = """
import math
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import numpy as np
import os


def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    x = int(f * 10 ** n)
    return float(x)/(10**n)


def read_file(file_name):
    x, y, q, a, b, serv = [], [], [], [], [], []
    K, Q = 0, 0
    with open (file_name) as f:
        lines = f.readlines()
        N = len(lines) - 10
        K,Q = lines[4].split()
        K = int(K)
        Q = int(Q)
        for i in range(9, 10 + N):
            _,x_i, y_i, q_i, a_i, b_i, serv_i  = str(lines[i]).split()
            x.append(int(x_i))
            y.append(int(y_i))
            q.append(int(q_i))
            a.append(int(a_i))
            b.append(int(b_i))
            serv.append(int(serv_i))
    return K, Q, x, y, q, a, b, serv

def distance(x1, y1, x2, y2):
    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return dist

# Distance callback

class CreateDistanceCallback(object):
  # Create callback to calculate distances and travel times between points.

  def __init__(self, locations):
    # Initialize distance array.
    num_locations = len(locations)
    self.matrix = {}

    for from_node in range(num_locations):
      self.matrix[from_node] = {}
      for to_node in range(num_locations):
        x1 = locations[from_node][0]
        y1 = locations[from_node][1]
        x2 = locations[to_node][0]
        y2 = locations[to_node][1]
        self.matrix[from_node][to_node] = distance(x1, y1, x2, y2)

  def Distance(self, from_node, to_node):
    return self.matrix[from_node][to_node]


# Demand callback
class CreateDemandCallback(object):
  # Create callback to get demands at location node.

  def __init__(self, demands):
    self.matrix = demands

  def Demand(self, from_node, to_node):
    return self.matrix[from_node]

# Service time (proportional to demand) callback.
class CreateServiceTimeCallback(object):
  # Create callback to get time windows at each location.

  def __init__(self, service_times):
    self.matrix = service_times

  def ServiceTime(self, from_node, to_node):
    return self.matrix[from_node]
# Create the travel time callback (equals distance divided by speed).
class CreateTravelTimeCallback(object):
  # Create callback to get travel times between locations.

  def __init__(self, dist_callback):
    self.dist_callback = dist_callback

  def TravelTime(self, from_node, to_node):
    travel_time = self.dist_callback(from_node, to_node)
    return travel_time
# Create total_time callback (equals service time plus travel time).
class CreateTotalTimeCallback(object):
  # Create callback to get total times between locations.

  def __init__(self, service_time_callback, travel_time_callback):
    self.service_time_callback = service_time_callback
    self.travel_time_callback = travel_time_callback

  def TotalTime(self, from_node, to_node):
    service_time = self.service_time_callback(from_node, to_node)
    travel_time = self.travel_time_callback(from_node, to_node)
    return service_time + travel_time
def main(file_name):
  # Create the data.
  data = create_data_array(file_name)
  locations = data[0]
  demands = data[1]
  start_times = data[2]
  end_times = data[3]
  service_times = data[4]
  Q = data[5]
  K = data[6]
  num_locations = len(locations)
  depot = 0
  num_vehicles = K
  search_time_limit = 1000

  # Create routing model.
  if num_locations > 0:
    print(num_locations)
    print(num_vehicles)
    print(demands)
    print(Q)

    # The number of nodes of the VRP is num_locations.
    # Nodes are indexed from 0 to num_locations - 1. By default the start of
    # a route is node 0.
    routing = pywrapcp.RoutingModel(num_locations, num_vehicles, depot)
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()

    # Setting first solution heuristic: the
    # method for finding a first solution to the problem.
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
    search_parameters.time_limit_ms = search_time_limit * 1000

    # search_parameters.local_search_operators.use_path_lns = False
    # search_parameters.local_search_operators.use_inactive_lns = False

    # The 'PATH_CHEAPEST_ARC' method does the following:
    # Starting from a route "start" node, connect it to the node which produces the
    # cheapest route segment, then extend the route by iterating on the last
    # node added to the route.

    # Put callbacks to the distance function and travel time functions here.

    dist_between_locations = CreateDistanceCallback(locations)
    dist_callback = dist_between_locations.Distance

    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
    demands_at_locations = CreateDemandCallback(demands)
    demands_callback = demands_at_locations.Demand

    # Adding capacity dimension constraints.
    VehicleCapacity = Q;
    NullCapacitySlack = 0;
    fix_start_cumul_to_zero = True
    capacity = "Capacity"

    routing.AddDimension(demands_callback, NullCapacitySlack, VehicleCapacity,
                         fix_start_cumul_to_zero, capacity)
    # Add time dimension.
    horizon = 24 * 3600
    time = "Time"

    service_times = CreateServiceTimeCallback(service_times)
    service_time_callback = service_times.ServiceTime

    travel_times = CreateTravelTimeCallback(dist_callback)
    travel_time_callback = travel_times.TravelTime

    total_times = CreateTotalTimeCallback(service_time_callback, travel_time_callback)
    total_time_callback = total_times.TotalTime

    routing.AddDimension(total_time_callback,  # total time function callback
                         horizon,
                         horizon,
                         fix_start_cumul_to_zero,
                         time)
    # Add time window constraints.
    time_dimension = routing.GetDimensionOrDie(time)

    for location in range(1, num_locations):
      start = start_times[location]
      end = end_times[location]
      time_dimension.CumulVar(location).SetRange(start, end)
    # Solve displays a solution if any.
    # print("here")
    assignment = routing.SolveWithParameters(search_parameters)
    # print("there")
    if assignment:
      size = len(locations)
      # Solution cost.
      print ("Total distance of all routes: " + str(assignment.ObjectiveValue()) + "\\n")
      # Inspect solution.
      capacity_dimension = routing.GetDimensionOrDie(capacity);
      time_dimension = routing.GetDimensionOrDie(time);

      total_cost = 0

      sol = []
      for vehicle_nbr in range(num_vehicles):
        index = routing.Start(vehicle_nbr)
        plan_output = 'Route {0}:'.format(vehicle_nbr)

        while not routing.IsEnd(index):
        #   print("running...")
          node_index = routing.IndexToNode(index)
          load_var = capacity_dimension.CumulVar(index)
          time_var = time_dimension.CumulVar(index)
          plan_output += \\
                    " {node_index} Load({load}) Time({tmin}, {tmax}) -> ".format(
                        node_index=node_index,
                        load=assignment.Value(load_var),
                        tmin=str(assignment.Min(time_var)),
                        tmax=str(assignment.Max(time_var)))
          index2 = assignment.Value(routing.NextVar(index))
          ni = routing.IndexToNode(index)
          ni2 = routing.IndexToNode(index2)
          total_cost = total_cost + truncate(distance(locations[ni][0], locations[ni][1], locations[ni2][0], locations[ni2][1]),1)
          sol.append((ni,ni2))
          index = index2
        node_index = routing.IndexToNode(index)
        load_var = capacity_dimension.CumulVar(index)
        time_var = time_dimension.CumulVar(index)
        plan_output += \\
                  " {node_index} Load({load}) Time({tmin}, {tmax})".format(
                      node_index=node_index,
                      load=assignment.Value(load_var),
                      tmin=str(assignment.Min(time_var)),
                      tmax=str(assignment.Max(time_var)))
        # print (plan_output)
        # print ("\\n")
      print(total_cost)
      result_out = "../Result_in_out/%s/" + os.path.basename(file_name)
      with open(result_out, "w") as ff:
          ff.write(str(total_cost) + "\\n")
      file_out = "../Test_in_out/%s/" + os.path.basename(file_name)
      with open(file_out, "w") as f:
          for arc in sol:
              l1 = locations[arc[0]]
              l2 = locations[arc[1]]
              if not (l1[0] == l2[0] and l1[1] == l2[1]):
                  f.write(str(l1[0]) + " " + str(l1[1]) + " " + str(l2[0]) + " "+ str(l2[1]) + "\\n")
    else:
      print ('No solution found.')
  else:
    print ('Specify an instance greater than 0.')

def create_data_array(file_name):

  K, Q, x, y, q, a, b, serv = read_file(file_name)

  locations = [[x[i], y[i]] for i in range(len(x))]

  demands =  q

  start_times =  a

  # tw_duration is the width of the time windows.

  # In this example, the width is the same at each location, so we define the end times to be
  # start times + tw_duration. For problems in which the time window widths vary by location,
  # you can explicitly define the list of end_times, as we have done for start_times.
  end_times = b

  service_times = serv
  data = [locations, demands, start_times, end_times, service_times, Q, K]
  return data
if __name__ == '__main__':
  # main("/Users/duynguyen/Documents/Internship/code/Test/rc101")
  done = []
  for file in os.listdir("../Test_in_out/%s/"):
      done.append(file)
  i = 0
  for file in os.listdir("../Test_in/%s/"):
      if file.startswith("%s") and not file in done:
          i += 1
          if i %s %s == %s:
              print("################################--" + file + "--################################")
              main("../Test_in/%s/" + file)
  print("done!")
"""
for i in range(0,16):
    f = "or-Solver-" + str(i) + ".py"
    with open(f, "w") as f:
        f.write(towrite % (str(i//8 + 5),str(i//8 + 5),\
        str(i//8 + 5), str(i//8 + 5), "rc", "%",str(8), str(i % 8), str(i//8 + 5)))
        # f.write(towrite)
