from multiprocessing import Process, Queue, cpu_count
import time

class DistributedTaskQueue:
    def __init__(self, num_workers=None):
        """
        Initializes the task queue.
        Args:
            num_workers (int): Number of worker processes. Defaults to the CPU count.
        """
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.num_workers = num_workers or cpu_count()

    def worker(self):
        """
        Worker function to process tasks from the queue.
        """
        while not self.task_queue.empty():
            task = self.task_queue.get()
            try:
                result = task["func"](*task["args"], **task["kwargs"])
                self.result_queue.put({"task_id": task["task_id"], "result": result})
            except Exception as e:
                self.result_queue.put({"task_id": task["task_id"], "error": str(e)})

    def add_task(self, func, *args, **kwargs):
        """
        Adds a task to the queue.
        Args:
            func (callable): The function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
        """
        task_id = f"task_{time.time_ns()}"
        self.task_queue.put({"task_id": task_id, "func": func, "args": args, "kwargs": kwargs})

    def process_tasks(self):
        """
        Distributes tasks across worker processes.
        """
        processes = [Process(target=self.worker) for _ in range(self.num_workers)]
        for process in processes:
            process.start()
        for process in processes:
            process.join()

    def get_results(self):
        """
        Retrieves all results from the result queue.
        Returns:
            list: List of task results.
        """
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())
        return results

# Example usage:
# def sample_task(x, y):
#     return x + y

# queue = DistributedTaskQueue(num_workers=4)
# for i in range(10):
#     queue.add_task(sample_task, i, i * 2)
# queue.process_tasks()
# print(queue.get_results())
