from fran.callback.case_recorder import CaseIDRecorderSnapshot


class CaseIDRecorderSnapshotDet(CaseIDRecorderSnapshot):
    def __init__(
        self,
        freq=5,
        local_folder="/tmp",
        dpi=300,
        monitor_dl="valid",
        dl_idx=0,
        labels_for_plots: list | tuple = (1,),
    ):
        super().__init__(
            vip_label=1,
            freq=freq,
            local_folder=local_folder,
            dpi=dpi,
            labels_for_plots=labels_for_plots,
            monitor_dl=monitor_dl,
            dl_idx=dl_idx,
        )
