import tqdm

from video_dataset import VideoDataset

class FullVideosFeaturesDataset():
    def __init__(self, dataset: VideoDataset, transform=None, verbose=True):
        self.dataset = dataset
        self.transform = transform
        self.verbose = verbose
        
        self.videos_segments_indices = self.__get_videos_segments_indices()
    
    def __get_videos_segments_indices(self):
        if hasattr(self.dataset, '_cached_videos_segments_indices'):
            return self.dataset._cached_videos_segments_indices

        videos_segments_indices = {}
        iterator = tqdm.tqdm(range(len(self.dataset))) if self.verbose else range(len(self.dataset))
        
        for sample_index in iterator:
            _, _, video_id, segment_index = self.dataset[sample_index]
            
            if video_id not in videos_segments_indices:
                videos_segments_indices[video_id] = [sample_index]
            else:
                videos_segments_indices[video_id].append(sample_index)
        
        for video_id in videos_segments_indices:
            videos_segments_indices[video_id] = sorted(
                videos_segments_indices[video_id], 
                key=lambda sample_index: self.dataset[sample_index][3]
            )
        
        self.dataset._cached_videos_segments_indices = videos_segments_indices
        return videos_segments_indices
    
    def __len__(self):
        return len(self.videos_segments_indices.keys())
    
    def __getitem__(self, video_index):
        video_id = list(self.videos_segments_indices.keys())[video_index]
        video_segments_indices = self.videos_segments_indices[video_id]

        features = []
        annotations = []

        for sample_index in video_segments_indices:
            frames, annotation, _, _ = self.dataset[sample_index]
            features.append(frames)
            annotations.append(annotation)
            
        if self.transform is not None:
            return self.transform((features, annotations, video_id))
        else:
            return features, annotations, video_id