import os
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.manifold import TSNE
import umap
import plotly.express as px
from datasets import Dataset, DatasetDict
from bing_image_downloader import downloader
import zipfile
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

class CelebrityDatasetPipeline:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.celebrities = []
        self.setup_directories()

    def setup_directories(self):
        os.makedirs("celebrity_images", exist_ok=True)
        os.makedirs("face_crops", exist_ok=True)
        os.makedirs("dataset", exist_ok=True)

    def select_celebrities(self):
        self.celebrities = [
            # Actors
            {"id": 1, "name": "Johnny Depp", "category": "actor", "source": "hollywood"},
            {"id": 2, "name": "Leonardo DiCaprio", "category": "actor", "source": "hollywood"},
            {"id": 3, "name": "Jennifer Lawrence", "category": "actor", "source": "hollywood"},
            {"id": 4, "name": "Tom Cruise", "category": "actor", "source": "hollywood"},
            {"id": 12, "name": "Scarlett Johansson", "category": "actor", "source": "hollywood"},
            {"id": 13, "name": "Ryan Reynolds", "category": "actor", "source": "hollywood"},
            
            # Singers
            {"id": 5, "name": "Taylor Swift", "category": "singer", "source": "music"},
            {"id": 6, "name": "Ed Sheeran", "category": "singer", "source": "music"},
            {"id": 7, "name": "Beyonce", "category": "singer", "source": "music"},
            {"id": 14, "name": "Adele", "category": "singer", "source": "music"},
            {"id": 15, "name": "Justin Bieber", "category": "singer", "source": "music"},
            
            # Russian celebrities (relevant for VK Video)
            {"id": 8, "name": "Vladimir Putin", "category": "politician", "source": "russia"},
            {"id": 9, "name": "Khabib Nurmagomedov", "category": "sports", "source": "russia"},
            {"id": 16, "name": "Alla Pugacheva", "category": "singer", "source": "russia"},
            {"id": 17, "name": "Timati", "category": "singer", "source": "russia"},
            {"id": 18, "name": "Oksana Grigorieva", "category": "musician", "source": "russia"},
            # YouTubers
            {"id": 10, "name": "PewDiePie", "category": "youtuber", "source": "youtube"},
            {"id": 11, "name": "MrBeast", "category": "youtuber", "source": "youtube"},
            {"id": 19, "name": "Jake Paul", "category": "youtuber", "source": "youtube"},
            {"id": 20, "name": "Logan Paul", "category": "youtuber", "source": "youtube"},
        ]
        return self.celebrities

    def collect_images(self):
        collection_report = {}
        
        for celeb in tqdm(self.celebrities, desc="Collecting images"):
            try:
                # Create a specific directory for each celebrity
                download_path = os.path.join("celebrity_images", celeb['name'].replace(' ', '_'))
                os.makedirs(download_path, exist_ok=True)
                
                # Download up to 50 images for the celebrity
                downloader.download(celeb['name'], limit=50, output_dir="celebrity_images",
                                  adult_filter_off=False, force_replace=False, timeout=60, verbose=False)
                
                # Проверяем обе возможные структуры папок
                actual_download_path = os.path.join("celebrity_images", celeb['name'])
                if not os.path.exists(actual_download_path):
                    actual_download_path = download_path
                
                # Count the downloaded images
                image_files = [f for f in os.listdir(actual_download_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                collection_report[celeb['id']] = {
                    'name': celeb['name'],
                    'total_images': len(image_files),
                    'path': actual_download_path
                }
                print(f"Downloaded {len(image_files)} images for {celeb['name']}")
            except Exception as e:
                print(f"Error collecting images for {celeb['name']}: {e}")
                collection_report[celeb['id']] = {
                    'name': celeb['name'], 
                    'total_images': 0,
                    'path': None
                }
        
        return collection_report

    def detect_and_crop_faces(self, collection_report):
        # Initialize MTCNN detector on the chosen device
        detector = MTCNN(keep_all=True, thresholds=[0.6, 0.7, 0.7], 
                        min_face_size=20, device=self.device)
        
        processing_report = {}
        
        for celeb_id, celeb_info in tqdm(collection_report.items(), desc="Processing faces"):
            if celeb_info['total_images'] == 0 or celeb_info['path'] is None:
                processing_report[celeb_id] = {
                    'name': celeb_info['name'],
                    'processed_faces': 0,
                    'output_dir': None
                }
                continue
            
            # Create a directory for cropped faces of this celebrity
            celeb_output_dir = os.path.join("face_crops", f"celebrity_{celeb_id}")
            os.makedirs(celeb_output_dir, exist_ok=True)
            
            image_files = [f for f in os.listdir(celeb_info['path']) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            processed_faces = 0
            
            for img_file in image_files:
                img_path = os.path.join(celeb_info['path'], img_file)
                
                try:
                    # Open image and convert to RGB
                    image = Image.open(img_path).convert('RGB')
                    
                    # Добавить проверку на размер изображения
                    if max(image.size) < 50:  # Слишком маленькое изображение
                        continue
                    
                    # Detect faces in the image
                    boxes, probs = detector.detect(image)
                    
                    if boxes is not None:
                        # Iterate through detected faces
                        for i, (box, prob) in enumerate(zip(boxes, probs)):
                            # Apply confidence threshold
                            if prob > 0.9:
                                x1, y1, x2, y2 = map(int, box)
                                
                                # Добавить отступы и проверку границ
                                padding = 10
                                h, w = image.size[1], image.size[0]
                                x1 = max(0, x1 - padding)
                                y1 = max(0, y1 - padding)
                                x2 = min(w, x2 + padding)
                                y2 = min(h, y2 + padding)
                                
                                # Проверка на валидный размер кропа
                                if (x2 - x1) < 20 or (y2 - y1) < 20:
                                    continue
                                
                                # Crop the face region
                                face_crop = np.array(image)[y1:y2, x1:x2]
                                
                                # Ensure the crop is not empty
                                if face_crop.size > 0:
                                    # Create a unique filename for the face crop
                                    face_filename = f"{os.path.splitext(img_file)[0]}_face_{i}.jpg"
                                    face_path = os.path.join(celeb_output_dir, face_filename)
                                    # Save the cropped face
                                    cv2.imwrite(face_path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
                                    processed_faces += 1
                except Exception as e:
                    # Handle potential errors during image loading or processing
                    print(f"Error processing {img_path}: {e}")
                    continue
            
            processing_report[celeb_id] = {
                'name': celeb_info['name'],
                'processed_faces': processed_faces,
                'output_dir': celeb_output_dir
            }
            print(f"Extracted {processed_faces} faces for {celeb_info['name']}")
        
        return processing_report

    def build_dataset(self, processing_report):
        dataset_records = []
        MIN_SAMPLES_PER_CELEB = 5
        
        # Gather all processed face crops
        for celeb_id, process_info in processing_report.items():
            celeb_info = next((c for c in self.celebrities if c['id'] == celeb_id), None)
            if not celeb_info or not os.path.exists(process_info['output_dir']):
                continue
            
            face_files = [f for f in os.listdir(process_info['output_dir']) if f.endswith('.jpg')]
            
            # Фильтрация знаменитостей с малым количеством образцов
            if len(face_files) < MIN_SAMPLES_PER_CELEB:
                print(f"Skipping {celeb_info['name']} - only {len(face_files)} samples")
                continue
            
            for face_file in face_files:
                face_path = os.path.join(process_info['output_dir'], face_file)
                dataset_records.append({
                    'image_path': face_path,
                    'celebrity_id': celeb_id,
                    'celebrity_name': celeb_info['name'],
                    'category': celeb_info['category'],
                    'source': celeb_info['source'],
                    'filename': face_file
                })
        
        # Использование pandas для более удобного разделения
        df = pd.DataFrame(dataset_records)
        
        if len(df) == 0:
            print("Warning: No data available for dataset creation")
            return DatasetDict({
                'train': Dataset.from_list([]),
                'validation': Dataset.from_list([]), 
                'test': Dataset.from_list([])
            })
        
        # Стратифицированное разделение с использованием sklearn
        train_df, temp_df = train_test_split(
            df, test_size=0.2, stratify=df['celebrity_id'], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df['celebrity_id'], random_state=42
        )
        
        # Create DatasetDict
        dataset_dict = DatasetDict({
            'train': Dataset.from_pandas(train_df.reset_index(drop=True)),
            'validation': Dataset.from_pandas(val_df.reset_index(drop=True)), 
            'test': Dataset.from_pandas(test_df.reset_index(drop=True))
        })
        
        return dataset_dict

    def evaluate_quality(self, dataset):
        # Load pre-trained face recognition model
        recognizer = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((160, 160)), # InceptionResnetV1 expects 160x160 input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Создание кастомного датасета для батчевой обработки
        class FaceDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, transform=None):
                self.dataset = dataset
                self.transform = transform
                
            def __len__(self):
                return len(self.dataset)
                
            def __getitem__(self, idx):
                record = self.dataset[idx]
                image = Image.open(record['image_path']).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, record['celebrity_id'], record['celebrity_name']
        
        # Sample for evaluation to speed up the process
        sample_size = min(500, len(dataset)) # Limit sample size
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        sample_dataset = dataset.select(indices)
        
        face_dataset = FaceDataset(sample_dataset, transform=transform)
        dataloader = DataLoader(face_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        embeddings, labels, names = [], [], []
        
        # Extract embeddings с батчевой обработкой
        with torch.no_grad():
            for batch_images, batch_labels, batch_names in tqdm(dataloader, desc="Extracting embeddings"):
                batch_embeddings = recognizer(batch_images.to(self.device))
                embeddings.append(batch_embeddings.cpu().numpy())
                labels.extend(batch_labels.numpy())
                names.extend(batch_names)
        
        if len(embeddings) == 0:
            print("Warning: No embeddings extracted")
            return {
                'silhouette_score': 0,
                'neighbor_accuracy': 0,
                'total_samples_evaluated': 0,
                'unique_celebrities_evaluated': 0
            }
        
        embeddings = np.vstack(embeddings)
        labels = np.array(labels)
        
        # Visualization using UMAP
        reducer = umap.UMAP(n_components=2, random_state=42) # Reduce to 2 dimensions
        embeddings_2d = reducer.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 10))
        # Create scatter plot, coloring points by celebrity ID
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=labels, cmap='tab20', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('Face Embeddings Visualization (UMAP)')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.show()
        
        # Calculate metrics
        from sklearn.metrics import silhouette_score
        from sklearn.neighbors import NearestNeighbors
        
        # Silhouette score
        silhouette_avg = silhouette_score(embeddings, labels)
        
        # Neighbor accuracy
        nbrs = NearestNeighbors(n_neighbors=6).fit(embeddings) # Consider top 5 neighbors
        distances, indices = nbrs.kneighbors(embeddings)
        
        correct_neighbors = 0
        total_neighbors = 0
        
        for i, neighbor_indices in enumerate(indices):
            # Exclude the point itself (neighbor_indices[0])
            for neighbor_idx in neighbor_indices[1:]: 
                if labels[i] == labels[neighbor_idx]:
                    correct_neighbors += 1
                total_neighbors += 1
        
        neighbor_accuracy = correct_neighbors / total_neighbors if total_neighbors > 0 else 0
        
        return {
            'silhouette_score': silhouette_avg,
            'neighbor_accuracy': neighbor_accuracy,
            'total_samples_evaluated': len(embeddings),
            'unique_celebrities_evaluated': len(np.unique(labels))
        }

    def save_dataset(self, dataset, output_path="celebrity_faces_dataset"):
        # Save the dataset to disk
        dataset.save_to_disk(output_path)
        
        # Create a zip archive of the saved dataset
        zip_filename = f"{output_path}.zip"
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Create relative path for the archive
                    arcname = os.path.relpath(file_path, output_path)
                    zipf.write(file_path, arcname)
        
        print(f"Dataset saved to: {output_path}")
        print(f"Zip archive created: {zip_filename}")
        return zip_filename

    def run_pipeline(self):
        print("🎬 Starting Celebrity Dataset Pipeline...")
        
        # 1. Select celebrities
        print("\n1. Selecting celebrities...")
        celebrities = self.select_celebrities()
        print(f"   Selected {len(celebrities)} celebrities")
        
        # 2. Collect images
        print("\n2. Collecting images...")
        collection_report = self.collect_images()
        successful_collections = sum(1 for r in collection_report.values() if r['total_images'] >= 50) # Check for minimum images
        print(f"   Image collection initiated for {len(collection_report)} celebrities. "
              f"Successfully collected >= 50 images for {successful_collections} celebrities.")
        
        # 3. Detect and crop faces
        print("\n3. Detecting and cropping faces...")
        processing_report = self.detect_and_crop_faces(collection_report)
        total_faces_cropped = sum(r['processed_faces'] for r in processing_report.values())
        print(f"   Successfully detected and cropped {total_faces_cropped} faces.")
        
        # 4. Build dataset
        print("\n4. Building dataset...")
        dataset = self.build_dataset(processing_report)
        print(f"   Dataset created with:")
        for split, data in dataset.items():
            print(f"     - {split.capitalize()}: {len(data)} samples")
        
        # 5. Evaluate quality
        print("\n5. Evaluating quality of the training split...")
        metrics = self.evaluate_quality(dataset['train'])
        print(f"   Evaluation Metrics:")
        print(f"     - Silhouette Score: {metrics['silhouette_score']:.4f}")
        print(f"     - Neighbor Accuracy: {metrics['neighbor_accuracy']:.4f}")
        print(f"     - Total samples evaluated: {metrics['total_samples_evaluated']}")
        print(f"     - Unique celebrities evaluated: {metrics['unique_celebrities_evaluated']}")
        
        # 6. Save dataset
        print("\n6. Saving dataset...")
        output_zip_path = self.save_dataset(dataset)
        
        # Final report
        print("\n" + "="*50)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"📊 Final Statistics:")
        print(f"   • Celebrities included: {len(celebrities)}")
        print(f"   • Total face crops generated: {total_faces_cropped}")
        print(f"   • Dataset splits:")
        for split, data in dataset.items():
            print(f"     - {split.capitalize()}: {len(data)} samples")
        print(f"   • Quality Metrics (on training split):")
        print(f"     - Silhouette Score: {metrics['silhouette_score']:.4f}")
        print(f"     - Neighbor Accuracy: {metrics['neighbor_accuracy']:.4f}")
        print(f"📁 Final dataset archive: {output_zip_path}")
        
        return dataset, metrics

# Main execution block
if __name__ == "__main__":
    # Instantiate the pipeline
    pipeline = CelebrityDatasetPipeline()
    
    # Run the complete pipeline
    # This will perform all steps: selection, collection, processing, building, evaluation, and saving.
    final_dataset, final_metrics = pipeline.run_pipeline() 

    # Example of how to access the created dataset after running the pipeline:
    # print("\nAccessing the created dataset:")
    # print("Train dataset:", final_dataset['train'])
    # print("Validation dataset:", final_dataset['validation'])
    # print("Test dataset:", final_dataset['test'])

    # You can also load the saved dataset later:
    # from datasets import load_from_disk
    # loaded_dataset = load_from_disk("celebrity_faces_dataset")
    # print("\nLoaded dataset structure:", loaded_dataset)
