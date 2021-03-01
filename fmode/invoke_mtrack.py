from funcx.sdk.client import FuncXClient
import time

def invoke_mtrack(date, time, lon, lat):
    import mtrack
    mtrack.Mtrack.pipeline(date, time, lon, lat)


if (__name__ == '__main__'):

    fxc = FuncXClient()
    func_id = fxc.register_function(invoke_mtrack)

    f = open("..\track\quiet.txt", "r")
    try:
        line = f.readline()
        date, t, _, lon, _, lat, _, lon_index, lat_index, _, _ = line.split(",")

        task_id = fxc.run(date, t, lon, lat, endpoint_id="4f8d3973-24ae-4358-a79b-5526cb9529ae", function_id=func_id)

        while fxc.get_task(task_id)['pending'] == 'True':
            time.sleep(3)

        result = fxc.get_result(task_id)
        print("Task result", result)

    except Exception as e:
        print(e)