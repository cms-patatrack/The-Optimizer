from .fileManager import FileManager
class Optimizer:
  
  def __init__(self, file_manager) -> None:
    self.file_manager = file_manager if file_manager is not None else FileManager()
    pass

  def optimize(self):
    pass

      
  