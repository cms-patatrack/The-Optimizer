class FileManager:
    def __init__(self,
                 saving_enabled=True,
                 loading_enabled=False,
                 checkpoint_dir="tmp/checkpoint",
                 history_dir="tmp/history") -> None:
        self.saving_enabled = saving_enabled
        self.loading_enabled = loading_enabled
        self.checkpoint_dir = checkpoint_dir
        self.history_dir = history_dir
