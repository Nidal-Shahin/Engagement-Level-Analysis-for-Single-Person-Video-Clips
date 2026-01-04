# 0. Libraries
import gradio as gr
import cv2
import torch
import torch.nn as nn
import numpy as np
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tempfile
import os

# 1. Model Definition
class EngagementViT(nn.Module):
    def __init__(self, num_classes=4):
        super(EngagementViT, self).__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.feature_dim = 768
        self.gru = nn.GRU(input_size=self.feature_dim, hidden_size=256, num_layers=1, batch_first=True)
        self.ordinal_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        self.discriminator = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if len(x.shape) == 4:
            b, c, h, w = x.shape
            t = 1
            x = x.unsqueeze(1)
        else:
            b, t, c, h, w = x.shape
        x_reshaped = x.view(-1, c, h, w)
        features = self.backbone(x_reshaped) 
        features = features.view(b, t, -1) 
        gru_out, _ = self.gru(features)
        last_state = gru_out[:, -1, :] 
        return self.ordinal_head(last_state) / self.temperature, self.discriminator(last_state)

def logits_to_label(logits, device):
    thresholds = torch.tensor([0.25, 0.5, 0.75]).to(device).float()
    probs = torch.sigmoid(logits)
    return (probs > thresholds).sum(dim=1)

def get_color(level):
    colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0)]
    return colors[int(np.clip(level, 0, 3))]

# 2. Main Logic ---
def process_video(video_path, smooth_factor=5, batch_size=12, target_fps=5):
    if not video_path:
        return None, "Please upload a video first."
        
    MODEL_PATH = "best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EngagementViT()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    detector = cv2.FaceDetectorYN.create("face_detection_yunet.onnx", "", (0, 0))
    cap = cv2.VideoCapture(video_path)
    width, height, original_fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)

    if original_fps <= 0 or original_fps > 60:
        output_fps = 20.0 
    else:
        output_fps = original_fps

    skip_interval = max(1, int(output_fps / target_fps))
    temp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    out = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (width, height))

    transform = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])
    
    frames_data = []
    tensors_to_process = []
    indices_in_inference_list = []
    all_predictions = []
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        face_info = None
        if frame_count % skip_interval == 0:
            detector.setInputSize((width, height))
            _, faces = detector.detect(frame)
            if faces is not None:
                bbox = faces[0][0:4].astype(int)
                x, y, w, h = bbox
                face_img = frame[max(0, y):y+h, max(0, x):x+w]
                if face_img.size > 0:
                    tensor = transform(image=cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))["image"]
                    face_info = {"bbox": bbox, "tensor": tensor}
                    tensors_to_process.append(tensor)
                    indices_in_inference_list.append(frame_count)
        
        frames_data.append({"frame": frame, "face": face_info, "engagement": 0})
        frame_count += 1

        if len(tensors_to_process) >= batch_size:
            with torch.no_grad():
                logits, _ = model(torch.stack(tensors_to_process).to(device))
                preds = logits_to_label(logits, device).cpu().numpy()
                for p, f_idx in zip(preds, indices_in_inference_list):
                    frames_data[f_idx]["engagement"] = p
                    all_predictions.append(p)
            tensors_to_process, indices_in_inference_list = [], []

    if tensors_to_process:
        with torch.no_grad():
            logits, _ = model(torch.stack(tensors_to_process).to(device))
            preds = logits_to_label(logits, device).cpu().numpy()
            for p, f_idx in zip(preds, indices_in_inference_list):
                frames_data[f_idx]["engagement"] = p
                all_predictions.append(p)

    engagement_scores = [f["engagement"] for f in frames_data]
    for i in range(len(engagement_scores)):
        if i % skip_interval != 0:
            engagement_scores[i] = engagement_scores[i - (i % skip_interval)]

    smoothed = np.convolve(engagement_scores, np.ones(smooth_factor)/smooth_factor, mode='same')

    for i, data in enumerate(frames_data):
        frame = data["frame"]
        anchor_idx = i - (i % skip_interval)
        anchor_face = frames_data[anchor_idx]["face"]
        if anchor_face is not None:
            x, y, w, h = anchor_face["bbox"]
            color = get_color(smoothed[i])
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 6)
            cv2.putText(frame, f"Eng: {smoothed[i]:.1f}", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
        out.write(frame)

    cap.release()
    out.release()

    L = np.mean(all_predictions) if all_predictions else 0
    percentage = (L / 3.0) * 100
    
    if L >= 2.5: level_text = "Very High Engagement"
    elif L >= 1.5: level_text = "High Engagement"
    elif L >= 0.5: level_text = "Low Engagement"
    else: level_text = "Very Low Engagement"

    results = f"Score: {L:.2f}/3.0\nPercentage: {percentage:.1f}%\nLevel: {level_text}"
    return temp_out, results

# 3. Gradio-Based UI 
with gr.Blocks() as demo:
    gr.Markdown("# Person Engagement Analysis")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Input Video")
            with gr.Accordion("Advanced Settings", open=False):
                batch_slider = gr.Slider(1, 32, value=12, step=1, label="Batch Size (Higher = Faster overall, but risks OOM)")
                smooth_slider = gr.Slider(1, 20, value=5, step=1, label="Smoothness (Higher = Less flickering)")
                fps_slider = gr.Slider(1, 30, value=5, step=1, label="Analysis FPS (Higher = Slower inference, but better details)")
            submit_btn = gr.Button("Analyze Engagement", variant="primary")
            
        with gr.Column():
            video_output = gr.Video(label="Processed Video")
            results_text = gr.Textbox(label="Overall Engagement Results", lines=3)

    # Test Examples Section
    gr.Markdown("### Test Examples")
    gr.Examples(
        examples=[
            ["test_samples/Class_0_Example.mp4", 5, 12, 5],
            ["test_samples/Class_1_Example.mp4", 5, 12, 5],
            ["test_samples/Class_2_Example.mp4", 5, 12, 5],
            ["test_samples/Class_3_Example.mp4", 5, 12, 5]
        ],
        inputs=[video_input, smooth_slider, batch_slider, fps_slider],
        outputs=[video_output, results_text],
        fn=process_video,
        cache_examples=False,
        label="Select a pre-uploaded example to analyze with default settings"
    )
    
    submit_btn.click(
        process_video, 
        inputs=[video_input, smooth_slider, batch_slider, fps_slider], 
        outputs=[video_output, results_text]
    )

demo.launch(allowed_paths=["test_samples"])