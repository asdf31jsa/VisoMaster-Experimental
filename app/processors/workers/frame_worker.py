import traceback
from typing import TYPE_CHECKING
import threading
import math
from math import floor, ceil
from PIL import Image
import torch
from skimage import transform as trans
import kornia.enhance as ke
import kornia.color as kc

import os

from torchvision.transforms import v2
import torchvision
from torchvision import transforms
from torchvision import models
from torchvision.transforms.functional import normalize

import numpy as np
import cv2
import torch.nn.functional as F

from app.processors.models_data import models_dir # For UNet model existence check

from app.processors.utils import faceutil
import app.ui.widgets.actions.common_actions as common_widget_actions
from app.ui.widgets.actions import video_control_actions
from app.helpers.miscellaneous import ParametersDict, get_scaling_transforms

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

torchvision.disable_beta_transforms_warning()
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

t512, t384, t256, t128, interpolation_get_cropped_face_kps, interpolation_original_face_128_384, interpolation_original_face_512, interpolation_Untransform, t256_face, interpolation_expression_faceeditor_back, interpolation_block_shift = None, None, None, None, None, None, None, None, None, None, None
class FrameWorker(threading.Thread):
    def __init__(self, frame, main_window: 'MainWindow', frame_number, frame_queue, is_single_frame=False):
        super().__init__()
        self.frame_queue = frame_queue
        self.frame = frame
        self.main_window = main_window
        self.frame_number = frame_number
        self.models_processor = main_window.models_processor
        self.video_processor = main_window.video_processor
        self.is_single_frame = is_single_frame
        self.parameters = {}
        self.target_faces = main_window.target_faces
        self.compare_images = []
        self.is_view_face_compare: bool = False
        self.is_view_face_mask: bool = False
        self.lock = threading.Lock()
    
    def run(self):
        try:
            # Update parameters from markers (if exists) without concurrent access from other threads
            with self.main_window.models_processor.model_lock:
                video_control_actions.update_parameters_and_control_from_marker(self.main_window, self.frame_number)
            self.parameters = self.main_window.parameters.copy()
            # Check if view mask or face compare checkboxes are checked
            self.is_view_face_compare = self.main_window.faceCompareCheckBox.isChecked() 
            self.is_view_face_mask = self.main_window.faceMaskCheckBox.isChecked() 

            # Process the frame with model inference
            # print(f"Processing frame {self.frame_number}")
            if self.main_window.swapfacesButton.isChecked() or self.main_window.editFacesButton.isChecked() or self.main_window.control['FrameEnhancerEnableToggle']:
                self.frame = self.process_frame()
            else:
                # Img must be in BGR format
                self.frame = self.frame[..., ::-1]  # Swap the channels from RGB to BGR
            self.frame = np.ascontiguousarray(self.frame)

            # Display the frame if processing is still active

            pixmap = common_widget_actions.get_pixmap_from_frame(self.main_window, self.frame)

            # Output processed Webcam frame
            if self.video_processor.file_type=='webcam' and not self.is_single_frame:
                self.video_processor.webcam_frame_processed_signal.emit(pixmap, self.frame)

            #Output Video frame (while playing)
            elif not self.is_single_frame:
                self.video_processor.frame_processed_signal.emit(self.frame_number, pixmap, self.frame)
            # Output Image/Video frame (Single frame)
            else:
                # print('Emitted single_frame_processed_signal')
                self.video_processor.single_frame_processed_signal.emit(self.frame_number, pixmap, self.frame)


            # Mark the frame as done in the queue
            self.video_processor.frame_queue.get()
            self.video_processor.frame_queue.task_done()

            # Check if playback is complete
            if self.video_processor.frame_queue.empty() and not self.video_processor.processing and self.video_processor.next_frame_to_display >= self.video_processor.max_frame_number:
                self.video_processor.stop_processing()

        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"Error in FrameWorker: {e}")
            traceback.print_exc()
    def set_scaling_transforms(self, parameters):
        global t512, t384, t256, t128, interpolation_get_cropped_face_kps, interpolation_original_face_128_384, interpolation_original_face_512, interpolation_Untransform, t256_face, interpolation_expression_faceeditor_back, interpolation_block_shift  # Damit wir die globalen Variablen ändern können

        t512, t384, t256, t128, interpolation_get_cropped_face_kps, interpolation_original_face_128_384, interpolation_original_face_512, interpolation_Untransform, t256_face, interpolation_expression_faceeditor_back, interpolation_block_shift = get_scaling_transforms(parameters)    
    # @misc_helpers.benchmark
    
    def tensor_to_pil(self, tensor):
        # Falls Tensor eine Batch-Dimension hat (1, 3, 512, 512), entfernen
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # Falls Tensor nur eine Kanal-Dimension hat (1, 512, 512), umwandeln in 3-Kanal (grau → RGB)
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)  # Kopiert den Grauwert auf alle 3 Kanäle

        # Falls der Tensor float-Werte in [0,1] hat, skaliere auf [0,255]
        if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
            tensor = (tensor * 255).clamp(0, 255).byte()
            tensor = tensor.byte()
        # Kanalachsen von (C, H, W) → (H, W, C)
        tensor = tensor.permute(1, 2, 0).cpu().numpy()
        #print("tensor: ", tensor.shape)

        # PIL-Image erstellen
        return Image.fromarray(tensor)
        
    def process_frame(self):
        # Load frame into VRAM
        img = torch.from_numpy(self.frame.astype('uint8')).to(self.models_processor.device) #HxWxc
        img = img.permute(2,0,1)#cxHxW

        #Scale up frame if it is smaller than 512
        img_x = img.size()[2]
        img_y = img.size()[1]

        # det_scale = 1.0
        if img_x<512 and img_y<512:
            # if x is smaller, set x to 512
            if img_x <= img_y:
                new_height = int(512*img_y/img_x)
                tscale = v2.Resize((new_height, 512), antialias=False)
            else:
                new_height = 512
                tscale = v2.Resize((new_height, int(512*img_x/img_y)), antialias=False)

            img = tscale(img)

            # det_scale = torch.div(new_height, img_y)

        elif img_x<512:
            new_height = int(512*img_y/img_x)
            tscale = v2.Resize((new_height, 512), antialias=False)
            img = tscale(img)

            # det_scale = torch.div(new_height, img_y)

        elif img_y<512:
            new_height = 512
            tscale = v2.Resize((new_height, int(512*img_x/img_y)), antialias=False)
            img = tscale(img)

            # det_scale = torch.div(new_height, img_y)

        control = self.main_window.control.copy()
        # Rotate the frame
        if control['ManualRotationEnableToggle']:
            img = v2.functional.rotate(img, angle=control['ManualRotationAngleSlider'], interpolation=v2.InterpolationMode.BILINEAR, expand=True)

        use_landmark_detection=control['LandmarkDetectToggle']
        landmark_detect_mode=control['LandmarkDetectModelSelection']
        from_points = control["DetectFromPointsToggle"]
        if self.main_window.editFacesButton.isChecked():
            if not use_landmark_detection or landmark_detect_mode=="5":
                # force to use landmark detector when edit face is enabled.
                use_landmark_detection = True
                landmark_detect_mode = "203"

            # force to use from_points in landmark detector when edit face is enabled.
            from_points = True

        bboxes, kpss_5, kpss = self.models_processor.run_detect(img, control['DetectorModelSelection'], max_num=control['MaxFacesToDetectSlider'], score=control['DetectorScoreSlider']/100.0, input_size=(512, 512), use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=control["LandmarkDetectScoreSlider"]/100.0, from_points=from_points, rotation_angles=[0] if not control["AutoRotationToggle"] else [0, 90, 180, 270])
        
        det_faces_data = []
        if len(kpss_5)>0:
            for i in range(kpss_5.shape[0]):
                face_kps_5 = kpss_5[i]
                face_kps_all = kpss[i]
                face_emb, _ = self.models_processor.run_recognize_direct(img, face_kps_5, control['SimilarityTypeSelection'], control['RecognitionModelSelection'])
                det_faces_data.append({'kps_5': face_kps_5, 'kps_all': face_kps_all, 'embedding': face_emb, 'bbox': bboxes[i]})

        compare_mode = self.is_view_face_mask or self.is_view_face_compare
        
        if det_faces_data:
            for i, fface in enumerate(det_faces_data):
                # Flag: nur den besten Match swappen?
                best_only = control['SwapOnlyBestMatchEnableToggle']
                print("best_only: ", best_only)
                if best_only:
                    # ------------------
                    # Best-Only Modus
                    # ------------------
                    best_sim    = -1.0
                    best_target = None
                    best_params = None

                    for _, target_face in self.main_window.target_faces.items():
                        params = ParametersDict(
                            self.parameters[target_face.face_id],
                            self.main_window.default_parameters
                        )
                        self.set_scaling_transforms(params)
                        if not (self.main_window.swapfacesButton.isChecked()
                                or self.main_window.editFacesButton.isChecked()):
                            continue

                        sim = self.models_processor.findCosineDistance(
                            fface['embedding'],
                            target_face.get_embedding(control['RecognitionModelSelection'])
                        )
                        if sim >= params['SimilarityThresholdSlider'] and sim > best_sim:
                            best_sim    = sim
                            best_target = target_face
                            best_params = params

                    if best_target is not None:
                        # hier führst Du genau einen Swap mit best_target durch
                        parameters = best_params
                        arcface_model = self.models_processor.get_arcface_model(
                            parameters['SwapModelSelection']
                        )
                        dfm_model = parameters['DFMModelSelection']
                        s_e = None
                        if self.main_window.swapfacesButton.isChecked():
                            if parameters['SwapModelSelection'] != 'DeepFaceLive (DFM)':
                                s_e = best_target.assigned_input_embedding.get(arcface_model, None)
                            if s_e is not None and np.isnan(s_e).any():
                                s_e = None
                        img, fface['original_face'], fface['swap_mask'] = self.swap_core(
                            img, fface['kps_5'], fface['kps_all'],
                            s_e=s_e,
                            t_e=best_target.get_embedding(arcface_model),
                            parameters=parameters, control=control,
                            dfm_model=dfm_model
                        )
                        # ggf. Makeup
                        if (self.main_window.editFacesButton.isChecked()
                            and any(parameters[f] for f in (
                                'FaceMakeupEnableToggle',
                                'HairMakeupEnableToggle',
                                'EyeBrowsMakeupEnableToggle',
                                'LipsMakeupEnableToggle'
                            ))):
                            img = self.swap_edit_face_core_makeup(
                                img, fface['kps_all'], parameters, control
                            )

                else:
                    # ----------------------------------------
                    # Original-Modus: alle matches swappen
                    # ----------------------------------------
                    for _, target_face in self.main_window.target_faces.items():
                        params = ParametersDict(
                            self.parameters[target_face.face_id],
                            self.main_window.default_parameters
                        )
                        self.set_scaling_transforms(params)
                        if not (self.main_window.swapfacesButton.isChecked()
                                or self.main_window.editFacesButton.isChecked()):
                            continue

                        sim = self.models_processor.findCosineDistance(
                            fface['embedding'],
                            target_face.get_embedding(control['RecognitionModelSelection'])
                        )
                        if sim < params['SimilarityThresholdSlider']:
                            continue

                        # Keypoint-Anpassung
                        fface['kps_5'] = self.keypoints_adjustments(fface['kps_5'], params)
                        arcface_model = self.models_processor.get_arcface_model(
                            params['SwapModelSelection']
                        )
                        dfm_model = params['DFMModelSelection']
                        s_e = None
                        if self.main_window.swapfacesButton.isChecked():
                            if params['SwapModelSelection'] != 'DeepFaceLive (DFM)':
                                s_e = target_face.assigned_input_embedding.get(arcface_model, None)
                            if s_e is not None and np.isnan(s_e).any():
                                s_e = None

                        img, fface['original_face'], fface['swap_mask'] = self.swap_core(
                            img, fface['kps_5'], fface['kps_all'],
                            s_e=s_e,
                            t_e=target_face.get_embedding(arcface_model),
                            parameters=params, control=control,
                            dfm_model=dfm_model
                        )

                        if (self.main_window.editFacesButton.isChecked()
                            and any(params[f] for f in (
                                'FaceMakeupEnableToggle',
                                'HairMakeupEnableToggle',
                                'EyeBrowsMakeupEnableToggle',
                                'LipsMakeupEnableToggle'
                            ))):
                            img = self.swap_edit_face_core_makeup(
                                img, fface['kps_all'], params, control
                            )
                        
        if control['ManualRotationEnableToggle']:
            img = v2.functional.rotate(img, angle=-control['ManualRotationAngleSlider'], interpolation=v2.InterpolationMode.BILINEAR, expand=True)

        if control['ShowAllDetectedFacesBBoxToggle']:
            img = self.draw_bounding_boxes_on_detected_faces(img, det_faces_data, control)

        if control["ShowLandmarksEnableToggle"] and det_faces_data:
            img = img.permute(1,2,0)
            img = self.paint_face_landmarks(img, det_faces_data, control)
            img = img.permute(2,0,1)

        if compare_mode:
            img = self.get_compare_faces_image(img, det_faces_data, control)

        if control['FrameEnhancerEnableToggle'] and not compare_mode:
            img = self.enhance_core(img, control=control)
        
        if img_x < 512 or img_y < 512:
            tscale_back = v2.Resize((img_y, img_x), antialias=False)
            img = tscale_back(img)
        
        img = img.permute(1,2,0)
        img = img.cpu().numpy()
        # RGB to BGR
        return img[..., ::-1]
    
    def keypoints_adjustments(self, kps_5: np.ndarray, parameters: dict) -> np.ndarray:
        # Change the ref points
        if parameters['FaceAdjEnableToggle']:
            kps_5[:,0] += parameters['KpsXSlider']
            kps_5[:,1] += parameters['KpsYSlider']
            kps_5[:,0] -= 255
            kps_5[:,0] *= (1+parameters['KpsScaleSlider']/100)
            kps_5[:,0] += 255
            kps_5[:,1] -= 255
            kps_5[:,1] *= (1+parameters['KpsScaleSlider']/100)
            kps_5[:,1] += 255

        # Face Landmarks
        if parameters['LandmarksPositionAdjEnableToggle']:
            kps_5[0][0] += parameters['EyeLeftXAmountSlider']
            kps_5[0][1] += parameters['EyeLeftYAmountSlider']
            kps_5[1][0] += parameters['EyeRightXAmountSlider']
            kps_5[1][1] += parameters['EyeRightYAmountSlider']
            kps_5[2][0] += parameters['NoseXAmountSlider']
            kps_5[2][1] += parameters['NoseYAmountSlider']
            kps_5[3][0] += parameters['MouthLeftXAmountSlider']
            kps_5[3][1] += parameters['MouthLeftYAmountSlider']
            kps_5[4][0] += parameters['MouthRightXAmountSlider']
            kps_5[4][1] += parameters['MouthRightYAmountSlider']
        return kps_5
    
    def paint_face_landmarks(self, img: torch.Tensor, det_faces_data: list, control: dict) -> torch.Tensor:
        # if img_y <= 720:
        #     p = 1
        # else:
        #     p = 2
        p = 2 #Point thickness
        for i, fface in enumerate(det_faces_data):
            for _, target_face in self.main_window.target_faces.items():
                parameters = self.parameters[target_face.face_id] #Use the parameters of the target face
                sim = self.models_processor.findCosineDistance(fface['embedding'], target_face.get_embedding(control['RecognitionModelSelection']))
                if sim>=parameters['SimilarityThresholdSlider']:
                    if parameters['LandmarksPositionAdjEnableToggle']:
                        kcolor = tuple((255, 0, 0))
                        keypoints = fface['kps_5']
                    else:
                        kcolor = tuple((0, 255, 255))
                        keypoints = fface['kps_all']

                    for kpoint in keypoints:
                        for i in range(-1, p):
                            for j in range(-1, p):
                                try:
                                    img[int(kpoint[1])+i][int(kpoint[0])+j][0] = kcolor[0]
                                    img[int(kpoint[1])+i][int(kpoint[0])+j][1] = kcolor[1]
                                    img[int(kpoint[1])+i][int(kpoint[0])+j][2] = kcolor[2]

                                except ValueError:
                                    #print("Key-points value {} exceed the image size {}.".format(kpoint, (img_x, img_y)))
                                    continue
        return img
    
    def draw_bounding_boxes_on_detected_faces(self, img: torch.Tensor, det_faces_data: list, control: dict):
        for i, fface in enumerate(det_faces_data):
            color = [0, 255, 0]
            bbox = fface['bbox']
            x_min, y_min, x_max, y_max = map(int, bbox)
            # Ensure bounding box is within the image dimensions
            _, h, w = img.shape
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w - 1, x_max), min(h - 1, y_max)
            # Dynamically compute thickness based on the image resolution
            max_dimension = max(img.shape[1], img.shape[2])  # Height and width of the image
            thickness = max(4, max_dimension // 400)  # Thickness is 1/200th of the largest dimension, minimum 1
            # Prepare the color tensor with the correct dimensions
            color_tensor = torch.tensor(color, dtype=img.dtype, device=img.device).view(-1, 1, 1)
            # Draw the top edge
            img[:, y_min:y_min + thickness, x_min:x_max + 1] = color_tensor.expand(-1, thickness, x_max - x_min + 1)
            # Draw the bottom edge
            img[:, y_max - thickness + 1:y_max + 1, x_min:x_max + 1] = color_tensor.expand(-1, thickness, x_max - x_min + 1)
            # Draw the left edge
            img[:, y_min:y_max + 1, x_min:x_min + thickness] = color_tensor.expand(-1, y_max - y_min + 1, thickness)
            # Draw the right edge
            img[:, y_min:y_max + 1, x_max - thickness + 1:x_max + 1] = color_tensor.expand(-1, y_max - y_min + 1, thickness)   
        return img

    def get_compare_faces_image(self, img: torch.Tensor, det_faces_data: dict, control: dict) -> torch.Tensor:
        imgs_to_vstack = []  # Renamed for vertical stacking
        for _, fface in enumerate(det_faces_data):
            for _, target_face in self.main_window.target_faces.items():
                parameters = self.parameters[target_face.face_id]  # Use the parameters of the target face
                sim = self.models_processor.findCosineDistance(
                    fface['embedding'], 
                    target_face.get_embedding(control['RecognitionModelSelection'])
                )
                if sim >= parameters['SimilarityThresholdSlider']:
                    modified_face = self.get_cropped_face_using_kps(img, fface['kps_5'], parameters)
                    # Apply frame enhancer
                    if control['FrameEnhancerEnableToggle']:
                        # Enhance the face and resize it to the original size for stacking
                        modified_face_enhance = self.enhance_core(modified_face, control=control)
                        modified_face_enhance = modified_face_enhance.float() / 255.0
                        # Resize source_tensor to match the size of target_tensor
                        modified_face = torch.functional.F.interpolate(
                            modified_face_enhance.unsqueeze(0),  # Add batch dimension
                            size=modified_face.shape[1:],  # Target size: [H, W]
                            mode='bilinear',  # Interpolation mode
                            align_corners=False  # Avoid alignment artifacts
                        ).squeeze(0)  # Remove batch dimension
                        
                        modified_face = (modified_face * 255).clamp(0, 255).to(dtype=torch.uint8)
                    imgs_to_cat = []
                    
                    # Append tensors to imgs_to_cat
                    if fface['original_face'] is not None:
                        imgs_to_cat.append(fface['original_face'].permute(2, 0, 1))
                    imgs_to_cat.append(modified_face)
                    if fface['swap_mask'] is not None:
                        fface['swap_mask'] = 255-fface['swap_mask']
                        imgs_to_cat.append(fface['swap_mask'].permute(2, 0, 1))
  
                    # Concatenate horizontally for comparison
                    img_compare = torch.cat(imgs_to_cat, dim=2)

                    # Add horizontally concatenated image to vertical stack list
                    imgs_to_vstack.append(img_compare)
    
        if imgs_to_vstack:
            # Find the maximum width
            max_width = max(img_to_stack.size(2) for img_to_stack in imgs_to_vstack)
            
            # Pad images to have the same width
            padded_imgs = [
                torch.nn.functional.pad(img_to_stack, (0, max_width - img_to_stack.size(2), 0, 0)) 
                for img_to_stack in imgs_to_vstack
            ]
            # Stack images vertically
            img_vstack = torch.cat(padded_imgs, dim=1)  # Use dim=1 for vertical stacking
            img = img_vstack
        return img
        
    def get_cropped_face_using_kps(self, img: torch.Tensor, kps_5: np.ndarray, parameters: dict) -> torch.Tensor:
        tform = self.get_face_similarity_tform(parameters['SwapModelSelection'], kps_5)
        # Grab 512 face from image and create 256 and 128 copys
        face_512 = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0), interpolation=interpolation_get_cropped_face_kps)
        face_512 = v2.functional.crop(face_512, 0,0, 512, 512)# 3, 512, 512
        return face_512

    def get_face_similarity_tform(self, swapper_model: str, kps_5: np.ndarray) -> trans.SimilarityTransform:
        tform = trans.SimilarityTransform()
        if swapper_model != 'GhostFace-v1' and swapper_model != 'GhostFace-v2' and swapper_model != 'GhostFace-v3' and swapper_model != 'CSCS':
            dst = faceutil.get_arcface_template(image_size=512, mode='arcface128')
            dst = np.squeeze(dst)
            tform.estimate(kps_5, dst)
        elif swapper_model == "CSCS":
            dst = faceutil.get_arcface_template(image_size=512, mode='arcfacemap')
            tform.estimate(kps_5, self.models_processor.FFHQ_kps)
        else:
            dst = faceutil.get_arcface_template(image_size=512, mode='arcfacemap')
            M, _ = faceutil.estimate_norm_arcface_template(kps_5, src=dst)
            tform.params[0:2] = M
        return tform

    def get_transformed_and_scaled_faces(self, tform, img):
        # Grab 512 face from image and create 256 and 128 copys
        original_face_512 = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0), interpolation=interpolation_original_face_512)
        original_face_512 = v2.functional.crop(original_face_512, 0,0, 512, 512)# 3, 512, 512
        original_face_384 = t384(original_face_512)
        original_face_256 = t256(original_face_512)
        original_face_128 = t128(original_face_256)
        return original_face_512, original_face_384, original_face_256, original_face_128
    
    def get_affined_face_dim_and_swapping_latents(self, original_faces: tuple, swapper_model, dfm_model, s_e, t_e, parameters, tform):
        original_face_512, original_face_384, original_face_256, original_face_128 = original_faces
        if swapper_model == 'Inswapper128':
            self.models_processor.load_inswapper_iss_emap('Inswapper128')
            latent = torch.from_numpy(self.models_processor.calc_inswapper_latent(s_e)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle']:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_inswapper_latent(t_e)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)

            dim = 1
            
            if parameters['SwapperResAutoSelectEnableToggle']:
                if tform.scale <= 1.00:
                    dim = 4
                    input_face_affined = original_face_512
                    #print("Resolution = 512", tform.scale)
                elif tform.scale <= 1.75:
                    dim = 3
                    input_face_affined = original_face_384
                    #print("Resolution = 384", tform.scale)
                elif tform.scale <= 2:
                    dim = 2
                    input_face_affined = original_face_256
                    #print("Resolution = 256", tform.scale)
                else:
                    dim = 1
                    input_face_affined = original_face_128
                    #print("Resolution = 128", tform.scale)     
                if control["CommandLineDebugEnableToggle"]:
                    print("Resolution", 128*dim)#, tform.scale)   
            else:
                if parameters['SwapperResSelection'] == '128':
                    dim = 1
                    input_face_affined = original_face_128
                elif parameters['SwapperResSelection'] == '256':
                    dim = 2
                    input_face_affined = original_face_256
                elif parameters['SwapperResSelection'] == '384':
                    dim = 3
                    input_face_affined = original_face_384
                elif parameters['SwapperResSelection'] == '512':
                    dim = 4
                    input_face_affined = original_face_512

        elif swapper_model in ('InStyleSwapper256 Version A', 'InStyleSwapper256 Version B', 'InStyleSwapper256 Version C'):
            version = swapper_model[-1]
            self.models_processor.load_inswapper_iss_emap(swapper_model)
            latent = torch.from_numpy(self.models_processor.calc_swapper_latent_iss(s_e, version)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle']:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_swapper_latent_iss(t_e, version)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)
            
            if (parameters['SwapModelSelection'] == 'InStyleSwapper256 Version A' and parameters['InStyleResAEnableToggle']) or (parameters['SwapModelSelection'] == 'InStyleSwapper256 Version B' and parameters['InStyleResBEnableToggle']) or (parameters['SwapModelSelection'] == 'InStyleSwapper256 Version C' and parameters['InStyleResCEnableToggle']):
                dim = 4
                input_face_affined = original_face_512
            else:
                dim = 2
                input_face_affined = original_face_256

        elif swapper_model == 'SimSwap512':
            latent = torch.from_numpy(self.models_processor.calc_swapper_latent_simswap512(s_e)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle']:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_swapper_latent_simswap512(t_e)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)

            dim = 4
            input_face_affined = original_face_512

        elif swapper_model == 'GhostFace-v1' or swapper_model == 'GhostFace-v2' or swapper_model == 'GhostFace-v3':
            latent = torch.from_numpy(self.models_processor.calc_swapper_latent_ghost(s_e)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle']:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_swapper_latent_ghost(t_e)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)

            dim = 2
            input_face_affined = original_face_256

        elif swapper_model == 'CSCS':
            latent = torch.from_numpy(self.models_processor.calc_swapper_latent_cscs(s_e)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle']:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_swapper_latent_cscs(t_e)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)

            dim = 2
            input_face_affined = original_face_256

        elif swapper_model == 'DeepFaceLive (DFM)' and dfm_model:
            dfm_model = self.models_processor.load_dfm_model(dfm_model)
            latent = []
            input_face_affined = original_face_512
            dim = 4
        return input_face_affined, dfm_model, dim, latent
    
    def get_swapped_and_prev_face(self, output, input_face_affined, original_face_512, latent, itex, dim, swapper_model, dfm_model, parameters, ):
        # original_face_512, original_face_384, original_face_256, original_face_128 = original_faces
        
        if parameters['PreSwapSharpnessDecimalSlider'] != 1.0:
            input_face_affined = input_face_affined.permute(2, 0, 1)#.type(torch.uint8)
            input_face_affined = v2.functional.adjust_sharpness(input_face_affined, parameters['PreSwapSharpnessDecimalSlider'])

            input_face_affined = input_face_affined.permute(1, 2, 0)
        
        
        prev_face = input_face_affined.clone()
        if swapper_model == 'Inswapper128':
            with torch.no_grad():
                for _ in range(itex):
                    tiles = []

                    for j in range(dim):
                        for i in range(dim):
                            tile = input_face_affined[j::dim, i::dim]  # Raster-Stil
                            #print(f"Tile [{j},{i}] shape:", tile.shape)
                            tile = tile.permute(2, 0, 1)  # [C, H, W]
                            tiles.append(tile)

                    input_batch = torch.stack(tiles, dim=0).contiguous()  # [B, 3, 128, 128]
                    output_batch = torch.empty_like(input_batch)
                    #print("input_batch shape:", input_batch.shape)
                    #print("output_batch shape:", output_batch.shape)
                    idx = 0
                    for j in range(dim):
                        for i in range(dim):
                            input_tile = tiles[idx].unsqueeze(0).contiguous()  # [1, 3, 128, 128]
                            output_tile = torch.empty_like(input_tile)
                            self.models_processor.run_inswapper(input_tile, latent, output_tile)

                            output_tile = output_tile.squeeze(0).permute(1, 2, 0)  # [H, W, C]
                            output[j::dim, i::dim] = output_tile.clone()

                            idx += 1

                    prev_face = input_face_affined.clone()
                    input_face_affined = output.clone()

                output = torch.clamp(output * 255, 0, 255) 
                
        elif swapper_model in ('InStyleSwapper256 Version A',
                               'InStyleSwapper256 Version B',
                               'InStyleSwapper256 Version C'):
            version = swapper_model[-1]  # 'A'/'B'/'C'
            with torch.no_grad():
                dim_res = dim // 2
                output = torch.zeros_like(input_face_affined)

                for _ in range(itex):
                    # 1) Tile split: jedes Tile ist 256×256
                    tiles = []
                    for j in range(dim_res):
                        for i in range(dim_res):
                            tile = input_face_affined[j::dim_res, i::dim_res]       # [256,256,3]
                            tile = tile.permute(2, 0, 1).unsqueeze(0)       # [1,3,256,256]
                            tiles.append(tile.contiguous())

                    # 2) Inference je Tile
                    idx = 0
                    for j in range(dim_res):
                        for i in range(dim_res):
                            input_tile  = tiles[idx]
                            output_tile = torch.empty_like(input_tile)
                            # hier Dein run_iss_swapper call
                            self.models_processor.run_iss_swapper(
                                input_tile, latent, output_tile, version
                            )

                            # 3) zurück in HWC
                            out_tile = output_tile.squeeze(0).permute(1, 2, 0)  # [256,256,3]
                            # 4) wieder zusammenfügen
                            output[j::dim_res, i::dim_res] = out_tile
                            idx += 1

                    # 5) für nächsten Durchgang
                    prev_face           = input_face_affined.clone()
                    input_face_affined  = output.clone()

                # 6) skaliere zurück in [0,255]
                output = torch.clamp(output * 255, 0, 255)

        elif swapper_model == 'SimSwap512':
            for k in range(itex):
                input_face_disc = input_face_affined.permute(2, 0, 1)
                input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()
                swapper_output = torch.empty((1,3,512,512), dtype=torch.float32, device=self.models_processor.device).contiguous()
                self.models_processor.run_swapper_simswap512(input_face_disc, latent, swapper_output)
                swapper_output = torch.squeeze(swapper_output)
                swapper_output = swapper_output.permute(1, 2, 0)
                prev_face = input_face_affined.clone()
                input_face_affined = swapper_output.clone()

                output = swapper_output.clone()
                output = torch.mul(output, 255)
                output = torch.clamp(output, 0, 255)

        elif swapper_model == 'GhostFace-v1' or swapper_model == 'GhostFace-v2' or swapper_model == 'GhostFace-v3':
            for k in range(itex):
                input_face_disc = torch.mul(input_face_affined, 255.0).permute(2, 0, 1)
                input_face_disc = torch.div(input_face_disc.float(), 127.5)
                input_face_disc = torch.sub(input_face_disc, 1)
                #input_face_disc = input_face_disc[[2, 1, 0], :, :] # Inverte i canali da BGR a RGB (assumendo che l'input sia BGR)
                input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()
                swapper_output = torch.empty((1,3,256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()
                self.models_processor.run_swapper_ghostface(input_face_disc, latent, swapper_output, swapper_model)
                swapper_output = swapper_output[0]
                swapper_output = swapper_output.permute(1, 2, 0)
                swapper_output = torch.mul(swapper_output, 127.5)
                swapper_output = torch.add(swapper_output, 127.5)
                #swapper_output = swapper_output[:, :, [2, 1, 0]] # Inverte i canali da RGB a BGR (assumendo che l'input sia RGB)
                prev_face = input_face_affined.clone()
                input_face_affined = swapper_output.clone()
                input_face_affined = torch.div(input_face_affined, 255)

                output = swapper_output.clone()
                output = torch.clamp(output, 0, 255)

        elif swapper_model == 'CSCS':
            for k in range(itex):
                input_face_disc = input_face_affined.permute(2, 0, 1)
                input_face_disc = v2.functional.normalize(input_face_disc, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)
                input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()
                swapper_output = torch.empty((1,3,256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()
                self.models_processor.run_swapper_cscs(input_face_disc, latent, swapper_output)
                swapper_output = torch.squeeze(swapper_output)
                swapper_output = torch.add(torch.mul(swapper_output, 0.5), 0.5)
                swapper_output = swapper_output.permute(1, 2, 0)
                prev_face = input_face_affined.clone()
                input_face_affined = swapper_output.clone()

                output = swapper_output.clone()
                output = torch.mul(output, 255)
                output = torch.clamp(output, 0, 255)
        
        elif swapper_model == 'DeepFaceLive (DFM)' and dfm_model:
            out_celeb, _, _ = dfm_model.convert(original_face_512, parameters['DFMAmpMorphSlider']/100, rct=parameters['DFMRCTColorToggle'])
            prev_face = input_face_affined.clone()
            input_face_affined = out_celeb.clone()
            output = out_celeb.clone()

        output = output.permute(2, 0, 1)
        #if dim != 4 or swapper_model == 'DeepFaceLive (DFM)':
        swap = t512(output)
        #else:
        #   swap = output
        return swap, prev_face
    
    def get_border_mask(self, parameters):
        # Create border mask
        border_mask = torch.ones((128, 128), dtype=torch.float32, device=self.models_processor.device)
        border_mask = torch.unsqueeze(border_mask,0)

        # if parameters['BorderState']:
        top = parameters['BorderTopSlider']
        left = parameters['BorderLeftSlider']
        right = 128 - parameters['BorderRightSlider']
        bottom = 128 - parameters['BorderBottomSlider']

        border_mask[:, :top, :] = 0
        border_mask[:, bottom:, :] = 0
        border_mask[:, :, :left] = 0
        border_mask[:, :, right:] = 0

        border_mask_calc = border_mask.clone()

        gauss = transforms.GaussianBlur(parameters['BorderBlurSlider']*2+1, (parameters['BorderBlurSlider']+1)*0.2)
        border_mask = gauss(border_mask)
        return border_mask, border_mask_calc
            
    def swap_core(self, img, kps_5, kps=False, s_e=None, t_e=None, parameters=None, control=None, dfm_model=False): # img = RGB
        s_e = s_e if isinstance(s_e, np.ndarray) else []
        t_e = t_e if isinstance(t_e, np.ndarray) else []
        parameters = parameters or {}
        control = control or {}
        swapper_model = parameters['SwapModelSelection']

        tform = self.get_face_similarity_tform(swapper_model, kps_5)
        t512_mask = v2.Resize((512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        t384_mask = v2.Resize((384, 384), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        t256_mask = v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        t128_mask = v2.Resize((128, 128), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)

        original_face_512, original_face_384, original_face_256, original_face_128 = self.get_transformed_and_scaled_faces(tform, img)
        if parameters['AnalyseImageEnableToggle']:
            image_analyse = self.analyze_image(original_face_512)
            print("image_analyse", image_analyse)
        original_faces = (original_face_512, original_face_384, original_face_256, original_face_128)
        dim=1
        if (s_e is not None and len(s_e) > 0) or (swapper_model == 'DeepFaceLive (DFM)' and dfm_model):

            input_face_affined, dfm_model, dim, latent = self.get_affined_face_dim_and_swapping_latents(original_faces, swapper_model, dfm_model, s_e, t_e, parameters, tform)

            # Optional Scaling # change the transform matrix scaling from center
            if parameters['FaceAdjEnableToggle']:
                input_face_affined = v2.functional.affine(input_face_affined, 0, (0, 0), 1 + parameters['FaceScaleAmountSlider'] / 100, 0, center=(dim*128/2, dim*128/2), interpolation=v2.InterpolationMode.BILINEAR)

            itex = 1
            if parameters['StrengthEnableToggle']:
                itex = ceil(parameters['StrengthAmountSlider'] / 100.)

            # Create empty output image and preprocess it for swapping
            output_size = int(128 * dim)
            output = torch.zeros((output_size, output_size, 3), dtype=torch.float32, device=self.models_processor.device)
            input_face_affined = input_face_affined.permute(1, 2, 0)
            input_face_affined = torch.div(input_face_affined, 255.0)

            swap, prev_face = self.get_swapped_and_prev_face(output, input_face_affined, original_face_512, latent, itex, dim, swapper_model, dfm_model, parameters)
        
        else:
            swap = original_face_512
            if parameters['StrengthEnableToggle']:
                itex = ceil(parameters['StrengthAmountSlider'] / 100.)
                prev_face = torch.div(swap, 255.)
                prev_face = prev_face.permute(1, 2, 0)

        if parameters['StrengthEnableToggle']:
            if itex == 0:
                swap = original_face_512.clone()
            else:
                alpha = np.mod(parameters['StrengthAmountSlider'], 100)*0.01
                if alpha==0:
                    alpha=1

                # Blend the images
                prev_face = torch.mul(prev_face, 255)
                prev_face = torch.clamp(prev_face, 0, 255)
                prev_face = prev_face.permute(2, 0, 1)
                #if dim != 4:
                prev_face = t512(prev_face)
                swap = torch.mul(swap, alpha)
                prev_face = torch.mul(prev_face, 1-alpha)
                swap = torch.add(swap, prev_face)

        # Create masks
        border_mask, border_mask_calc = self.get_border_mask(parameters)
        swap_mask = torch.ones((128, 128), dtype=torch.float32, device=self.models_processor.device)
        swap_mask = torch.unsqueeze(swap_mask,0)
        calc_mask = torch.ones((256, 256), dtype=torch.float32, device=self.models_processor.device)
        calc_mask = torch.unsqueeze(calc_mask,0)
        
        BgExclude = torch.ones((512, 512), dtype=torch.float32, device=self.models_processor.device)
        BgExclude = torch.unsqueeze(BgExclude,0)
        diff_mask = BgExclude.clone()
        texture_mask_view = BgExclude.clone()     
        restore_mask = BgExclude.clone()
        texture_mask = BgExclude.clone()
        mask_forcalc = BgExclude.clone()
        mask_calc_dill = BgExclude.clone()
        FaceEditmaskOnes = swap_mask.clone()
        
        swap = torch.clamp(swap, 0.0, 255.0)        

        # Expression Restorer
        if parameters['FaceExpressionEnableToggle']:
            swap = self.apply_face_expression_restorer(original_face_512, swap, parameters)

        swap_original = swap.clone()   
            
        
        if parameters["FaceRestorerEnableToggle"]:
            swap_restorecalc = self.models_processor.apply_facerestorer(swap, parameters['FaceRestorerDetTypeSelection'], parameters['FaceRestorerTypeSelection'], parameters["FaceRestorerBlendSlider"], parameters['FaceFidelityWeightDecimalSlider'], control['DetectorScoreSlider'])                                    
        else:
            swap_restorecalc = swap.clone()
        
        # Occluder
        if parameters["OccluderEnableToggle"]:
            mask = self.models_processor.apply_occlusion(original_face_256, parameters["OccluderSizeSlider"])
            mask = t128_mask(mask)
            swap_mask = torch.mul(swap_mask, mask)
            gauss = transforms.GaussianBlur(parameters['OccluderXSegBlurSlider']*2+1, (parameters['OccluderXSegBlurSlider']+1)*0.2)
            swap_mask = gauss(swap_mask)
        
        mouth = 0
        BgExclude = 0
        BgExcludeOccluder = 0
        
        if parameters["FaceParserEnableToggle"] or (parameters["DFLXSegEnableToggle"] and parameters["DFLXSeg2EnableToggle"] and parameters["DFLXSegSizeSlider"] != parameters["DFLXSeg2SizeSlider"] and (parameters["DFLXSegBGEnableToggle"] or parameters["XSegMouthEnableToggle"])) or ((parameters["TransferTextureEnableToggle"] or parameters["DifferencingEnableToggle"]) and parameters["ExcludeMaskEnableToggle"]): #parameters["BgExcludeEnableToggle"]
            out = self.models_processor.process_masks_and_masks(
                swap_restorecalc,
                original_face_512,
                parameters
            )     
            # 2. Ziehe Dir aus dem Dict die Masken, die Du weiter brauchst:
            #BgExcludeRestorecalc  = out.get("BgExcludeRestorecalc", 1)
            BgExclude             = out.get("BgExclude",          0)
            #BgExcludeOccluder     = out.get("BgExcludeOccluder",   0)
            FaceParser_mask       = out.get("FaceParser_mask", 1)
            texture_mask          = out.get("texture_mask",    texture_mask)
            mouth                 = out.get("mouth",           0)        
            
            swap_mask       = FaceParser_mask * swap_mask
            
        # CLIPs
        if parameters["ClipEnableToggle"]:
            mask = self.models_processor.run_CLIPs(original_face_512, parameters["ClipText"], parameters["ClipAmountSlider"])
            mask = t128_mask(mask)
            swap_mask *= mask

        if parameters['RestoreMouthEnableToggle'] or parameters['RestoreEyesEnableToggle']:
            M = tform.params[0:2]
            ones_column = np.ones((kps_5.shape[0], 1), dtype=np.float32)
            homogeneous_kps = np.hstack([kps_5, ones_column])
            dst_kps_5 = np.dot(homogeneous_kps, M.T)

            img_swap_mask = torch.ones((1, 512, 512), dtype=torch.float32, device=self.models_processor.device).contiguous()
            img_orig_mask = torch.zeros((1, 512, 512), dtype=torch.float32, device=self.models_processor.device).contiguous()

            if parameters['RestoreMouthEnableToggle']:
                img_swap_mask = self.models_processor.restore_mouth(img_orig_mask, img_swap_mask, dst_kps_5, parameters['RestoreMouthBlendAmountSlider']/100, parameters['RestoreMouthFeatherBlendSlider'], parameters['RestoreMouthSizeFactorSlider']/100, parameters['RestoreXMouthRadiusFactorDecimalSlider'], parameters['RestoreYMouthRadiusFactorDecimalSlider'], parameters['RestoreXMouthOffsetSlider'], parameters['RestoreYMouthOffsetSlider'])
                img_swap_mask = torch.clamp(img_swap_mask, 0, 1)

            if parameters['RestoreEyesEnableToggle']:
                img_swap_mask = self.models_processor.restore_eyes(img_orig_mask, img_swap_mask, dst_kps_5, parameters['RestoreEyesBlendAmountSlider']/100, parameters['RestoreEyesFeatherBlendSlider'], parameters['RestoreEyesSizeFactorDecimalSlider'],  parameters['RestoreXEyesRadiusFactorDecimalSlider'], parameters['RestoreYEyesRadiusFactorDecimalSlider'], parameters['RestoreXEyesOffsetSlider'], parameters['RestoreYEyesOffsetSlider'], parameters['RestoreEyesSpacingOffsetSlider'])
                img_swap_mask = torch.clamp(img_swap_mask, 0, 1)

            gauss = transforms.GaussianBlur(parameters['RestoreEyesMouthBlurSlider']*2+1, (parameters['RestoreEyesMouthBlurSlider']+1)*0.2)
            img_swap_mask = gauss(img_swap_mask)

            img_swap_mask = t128_mask(img_swap_mask)
            swap_mask = torch.mul(swap_mask, img_swap_mask)
        
        if parameters["DFLXSegEnableToggle"]:           
            if parameters["DFLXSeg2EnableToggle"] and parameters["XSegMouthEnableToggle"] and parameters["DFLXSegSizeSlider"] != parameters["DFLXSeg2SizeSlider"]:
                mouth = mouth.unsqueeze(0)
                mouth = t256(mouth)
                #mouth_original = t256_mask(mouth_original)
                #mouth = torch.max(mouth, mouth_original)
            else:
                mouth = 0
            img_xseg = original_face_256
            
            img_mask, mask_forcalc, mask_calc_dill = self.models_processor.apply_dfl_xseg(img_xseg, -parameters["DFLXSegSizeSlider"], BgExcludeOccluder, mouth, parameters)
            mask_calc_dill = torch.mul(calc_mask, 1 - mask_calc_dill)
            calc_mask = torch.mul(calc_mask, 1 - mask_forcalc)

            img_mask = t128_mask(img_mask)
            swap_mask = torch.mul(swap_mask, 1 - img_mask)
        else:
            calc_mask = swap_mask.clone()

        calc_mask = t512_mask(calc_mask).clone()
        mask_calc_dill = t512_mask(mask_calc_dill).clone()
        
        calc_mask = torch.where(calc_mask > 0.1, 1, 0).float()
        mask_calc_dill = torch.where(mask_calc_dill > 0.1, 1, 0).float()
  
        if parameters["FaceRestorerEnableToggle"] and parameters["FaceRestorerAutoEnableToggle"]:

            original_face_512_autorestore = original_face_512.clone()
            swap_original_autorestore = swap_original.clone()
            alpha_restorer = float(parameters["FaceRestorerBlendSlider"])/100.0
            adjust_sharpness = float(parameters["FaceRestorerAutoSharpAdjustSlider"])
            scale_factor = round(tform.scale, 2)
            
            alpha_auto, blur_value = self.face_restorer_auto(original_face_512_autorestore, swap_original_autorestore, swap_restorecalc, alpha_restorer, adjust_sharpness, scale_factor, control["CommandLineDebugEnableToggle"], restore_mask)#, parameters["FaceRestorerMaskSlider"], parameters["AutoRestorerTenengradTreshSlider"]/100, parameters["AutoRestorerCombWeightSlider"]/100)

        if parameters["FaceRestorerAutoEnableToggle"] and parameters["FaceRestorerEnableToggle"]:
            if blur_value != 0:
                #kernel_size = 2 * blur_value + 1
                #sigma = blur_value * 0.2
                #swap = transforms.GaussianBlur(kernel_size, sigma)(swap_original) 
                swap = swap_original
                if control["CommandLineDebugEnableToggle"]:
                    print("blur: ", blur_value)
            elif alpha_auto != 0:
                swap = swap_restorecalc * alpha_auto + swap_original * (1 - alpha_auto)
            else:
                swap = swap_original 

        elif parameters["FaceRestorerEnableToggle"]:
            alpha_restorer = float(parameters["FaceRestorerBlendSlider"])/100.0
            swap = torch.add(torch.mul(swap_restorecalc, alpha_restorer), torch.mul(swap_original, 1 - alpha_restorer))                             
      
        swap_backup = swap.clone()
        
        if parameters["TransferTextureEnableToggle"] or parameters["DifferencingEnableToggle"] or parameters["AutoColorEnableToggle"]:
            mask = torch.zeros((512, 512), dtype=torch.uint8, device=self.models_processor.device)
            mask = mask.unsqueeze(0)
            
            mask_calc = mask_calc_dill.clone()
            mask_calc = 1 - mask_calc
            
            mask_calc = mask_calc + (BgExclude)
            mask_calc = torch.where(
                mask_calc > 0.01, 
                1,
                0
            )
            if parameters['BGExcludeBlurAmountSlider'] > 0:
                orig = mask_calc.clone()
                gauss = transforms.GaussianBlur(parameters['BGExcludeBlurAmountSlider']*2+1, (parameters['BGExcludeBlurAmountSlider']+1)*0.2)
                mask_calc = gauss(mask_calc.type(torch.float32))
                mask_calc = torch.max(mask_calc, orig) 

        if parameters["AutoColorEnableToggle"]:
            # --- AutoColor block ---
            if parameters['AutoColorTransferTypeSelection'] == 'Test_Mask' or parameters['AutoColorTransferTypeSelection'] == 'DFL_Orig':
                mask_autocolor = mask_calc.clone()
                mask_autocolor = (mask_autocolor > 0.1)

            swap_backup = swap.clone()
            
            if parameters['AutoColorTransferTypeSelection'] == 'Test':
                swap = faceutil.histogram_matching(original_face_512, swap, parameters["AutoColorBlendAmountSlider"])

            elif parameters['AutoColorTransferTypeSelection'] == 'Test_Mask':
                swap = faceutil.histogram_matching_withmask(original_face_512, swap, mask_autocolor, parameters["AutoColorBlendAmountSlider"])
                if parameters["ExcludeMaskEnableToggle"]:
                    swap_backup = faceutil.histogram_matching_withmask(original_face_512, swap_backup, mask_autocolor, parameters["AutoColorBlendAmountSlider"])
                
            elif parameters['AutoColorTransferTypeSelection'] == 'DFL_Test':
                swap = faceutil.histogram_matching_DFL_test(original_face_512, swap, parameters["AutoColorBlendAmountSlider"])

            elif parameters['AutoColorTransferTypeSelection'] == 'DFL_Orig':
                swap = faceutil.histogram_matching_DFL_Orig(original_face_512, swap, mask_autocolor, parameters["AutoColorBlendAmountSlider"])

        if parameters["TransferTextureEnableToggle"]:
            # --- TransferTexture block ---
            
            TransferTextureKernelSizeSlider = 12 #parameters['TransferTextureKernelSizeSlider']
            TransferTextureSigmaDecimalSlider = 4.00 # parameters['TransferTextureSigmaDecimalSlider']
            TransferTextureWeightSlider = 1 #parameters['TransferTextureWeightNewDecimalSlider']
            TransferTextureLambdSlider = 2 #8 #parameters['TransferTextureLambdSlider']
            TransferTexturePhiDecimalSlider = 9.7 #parameters['TransferTexturePhiDecimalSlider']
            TransferTextureGammaDecimalSlider = 0.5 #parameters['TransferTextureGammaDecimalSlider']
            TransferTextureThetaSlider = 1 #8 #parameters['TransferTextureThetaSlider']
            TextureFeatureLayerTypeSelection = 'combo_relu3_3_relu3_1' #parameters['TextureFeatureLayerTypeSelection']

            if parameters['TransferTextureClaheEnableToggle']:
                clip_limit = parameters['TransferTextureClipLimitDecimalSlider']
            else:
                clip_limit = 0.0
            
            alpha_clahe = parameters['TransferTextureAlphaClaheDecimalSlider']
            grid_size = (4,4) #(parameters['TransferTextureGridSizeSlider'], parameters['TransferTextureGridSizeSlider'])
            global_gamma = parameters['TransferTexturePreGammaDecimalSlider']
            global_contrast = parameters['TransferTexturePreContrastDecimalSlider']
                               
            diff_mask_texture = t128_mask(texture_mask.clone())
            
            #swap = torch.where(calc_mask, swap, original_face_512)
            texture_mask_view = calc_mask.clone()
            gradient_texture = self.gradient_magnitude(
                original_face_512, texture_mask_view,
                TransferTextureKernelSizeSlider, TransferTextureWeightSlider,
                TransferTextureSigmaDecimalSlider, TransferTextureLambdSlider,
                TransferTextureGammaDecimalSlider, TransferTexturePhiDecimalSlider,
                TransferTextureThetaSlider, clip_limit, alpha_clahe, grid_size, global_gamma, global_contrast
            )

            

            mask = torch.ones((128, 128), dtype=torch.uint8, device=self.models_processor.device)
            mask = mask.unsqueeze(0)
            mask_texture = t128_mask(calc_mask.clone())
            if parameters["ExcludeOriginalVGGMaskEnableToggle"]:
                swapped_face_resized = swap.clone()
                original_face_resized = original_face_512.clone()
                mask, diff_norm_texture = self.models_processor.apply_perceptual_diff_onnx(
                    swapped_face_resized, original_face_resized, mask_texture,
                    parameters['TextureLowerLimitThreshSlider']/100,
                    0,
                    parameters['TextureUpperLimitThreshSlider']/100,
                    parameters['TextureUpperLimitValueSlider']/100,
                    parameters['TextureMiddleLimitValueSlider']/100,
                    TextureFeatureLayerTypeSelection,
                    parameters['ExcludeVGGMaskEnableToggle']
                )
                if not parameters["ExcludeVGGMaskEnableToggle"]:                
                    mask = diff_norm_texture
                if parameters['TextureBlendAmountSlider'] > 0:                                    
                    gauss = transforms.GaussianBlur(parameters['TextureBlendAmountSlider']*2+1, (parameters['TextureBlendAmountSlider']+1)*0.2)
                    mask = gauss(mask.type(torch.float32)) 
           
            if parameters["ExcludeMaskEnableToggle"]:
                mask = mask + parameters["FaceParserBlendTextureSlider"]/100
                mask = mask.clamp(0.0, 1.0)
                diff_mask_texture = 1 - diff_mask_texture
                if parameters['FaceParserBlurTextureSlider'] > 0:
                    orig = diff_mask_texture.clone()
                    gauss = transforms.GaussianBlur(parameters['FaceParserBlurTextureSlider']*2+1, (parameters['FaceParserBlurTextureSlider']+1)*0.2)
                    diff_mask_texture = gauss(diff_mask_texture.type(torch.float32))
                    diff_mask_texture = torch.max(diff_mask_texture, orig)
                mask = mask * diff_mask_texture

                mask = mask.clamp(0.0, 1.0)
            elif parameters["ExcludeOriginalVGGMaskEnableToggle"]:
                mask = mask + (1-diff_mask_texture) 
            else:
                mask = 1 - mask

            mask = t512_mask(mask)                   

            mask = mask + (mask_calc)
            mask = mask.clamp(0.0, 1.0)

            swap_texture_backup = swap.clone()
            swap_texture_backup = faceutil.histogram_matching_DFL_Orig(original_face_512, swap_texture_backup, calc_mask, 100)

            gradient_texture = faceutil.histogram_matching_DFL_Orig(original_face_512, gradient_texture, calc_mask, 100)

            alpha = parameters['TransferTextureBlendAmountSlider'] /100   # 0 = kein Detail, 1 = voller Gradient
            w = alpha * (1 - mask)
            swap = swap_texture_backup * (1 - w) + gradient_texture * w

            texture_mask_view = mask.clone()
            swap = swap.clamp(0, 255)

        if parameters["DifferencingEnableToggle"]:
            # --- Differencing block ---

            swapped_face_resized = swap.clone()
            FeatureLayerTypeSelection = 'combo_relu3_3_relu3_1'#parameters['FeatureLayerTypeSelection']

            original_face_resized = original_face_512.clone()
            diff_mask = t128_mask(calc_mask.clone())

            lower_thresh  = parameters['DifferencingLowerLimitThreshSlider']  / 100.0
            upper_thresh  = parameters['DifferencingUpperLimitThreshSlider']  / 100.0
            middle_value  = parameters['DifferencingMiddleLimitValueSlider'] / 100.0
            upper_value   = parameters['DifferencingUpperLimitValueSlider'] / 100.0

            mask, diff_norm_texture = self.models_processor.apply_perceptual_diff_onnx(
                swapped_face_resized, original_face_resized, diff_mask,
                lower_thresh,
                0,
                upper_thresh,
                upper_value,
                middle_value,
                FeatureLayerTypeSelection,
                False
            )
        
            eps = 1e-6
            inv_lower = 1.0 / max(lower_thresh, eps)
            inv_mid   = 1.0 / max((upper_thresh - lower_thresh), eps)
            inv_high  = 1.0 / max((1.0 - upper_thresh), eps)

            # 3) Die drei linearen Teilstücke
            #    lower_value ist hier 0, daher entfällt das Offset
            res_low  = diff_norm_texture * inv_lower * middle_value
            res_mid  = middle_value + (diff_norm_texture - lower_thresh) * inv_mid * (upper_value - middle_value)
            res_high = upper_value  + (diff_norm_texture - upper_thresh) * inv_high * (1.0 - upper_value)

            # 4) Piecewise in zwei where-Aufrufen
            result = torch.where(
                diff_norm_texture < lower_thresh,
                res_low,                                             # Bereich [0, lower_thresh)
                torch.where(
                    diff_norm_texture > upper_thresh,
                    res_high,                                        # Bereich (upper_thresh, 1]
                    res_mid                                          # Bereich [lower_thresh, upper_thresh]
                )
            )
            mask = result                
            mask = t512_mask(mask)#(1-mask_calc_dill)# + (1-texture_mask)                                       
            #mask = mask.clamp(0.0, 1.0)            
            
            gauss = transforms.GaussianBlur(parameters['DifferencingBlendAmountSlider']*2+1, (parameters['DifferencingBlendAmountSlider']+1)*0.2)
            mask = gauss(mask.type(torch.float32))
            mask = mask + mask_calc
            mask = mask.clamp(0.0, 1.0) 
            swap = swap * mask + original_face_512 * (1 - mask)
            swap = swap.clamp(0, 255)   
            
            diff_mask = mask.clone()
 

        # Apply color corrections
        if parameters['ColorEnableToggle']:
            swap = torch.unsqueeze(swap,0).contiguous()
            swap = v2.functional.adjust_gamma(swap, parameters['ColorGammaDecimalSlider'], 1.0)
            swap = torch.squeeze(swap)
            swap = swap.permute(1, 2, 0).type(torch.float32)

            del_color = torch.tensor([parameters['ColorRedSlider'], parameters['ColorGreenSlider'], parameters['ColorBlueSlider']], device=self.models_processor.device)
            swap += del_color
            swap = torch.clamp(swap, min=0., max=255.)
            swap = swap.permute(2, 0, 1) / 255.0 #.type(torch.uint8)

            swap = v2.functional.adjust_brightness(swap, parameters['ColorBrightnessDecimalSlider'])
            swap = v2.functional.adjust_contrast(swap, parameters['ColorContrastDecimalSlider'])
            swap = v2.functional.adjust_saturation(swap, parameters['ColorSaturationDecimalSlider'])
            swap = v2.functional.adjust_sharpness(swap, parameters['ColorSharpnessDecimalSlider'])
            swap = v2.functional.adjust_hue(swap, parameters['ColorHueDecimalSlider'])
            
            swap = swap * 255.0
    
        if parameters['FaceEditorEnableToggle'] and self.main_window.editFacesButton.isChecked():
            editor_mask = t512_mask(swap_mask).clone()
            swap = swap * editor_mask + original_face_512 * (1 - editor_mask)
            swap = self.swap_edit_face_core(swap, kps, parameters, control)
            swap_mask = FaceEditmaskOnes
        # Restorer2
        if parameters["FaceRestorerEnable2Toggle"]:
            swap2 = self.models_processor.apply_facerestorer(swap, parameters['FaceRestorerDetType2Selection'], parameters['FaceRestorerType2Selection'], parameters["FaceRestorerBlend2Slider"], parameters['FaceFidelityWeight2DecimalSlider'], control['DetectorScoreSlider'])
            alpha_restorer2 = float(parameters["FaceRestorerBlend2Slider"])/100.0
            swap = torch.add(torch.mul(swap2, alpha_restorer2), torch.mul(swap, 1 - alpha_restorer2))                            

        if parameters['FinalBlendAdjEnableToggle'] and parameters['FinalBlendAmountSlider'] > 0:
            final_blur_strength = parameters['FinalBlendAmountSlider']  # Ein Parameter steuert beides
            # Bestimme kernel_size und sigma basierend auf dem Parameter
            kernel_size = 2 * final_blur_strength + 1  # Ungerade Zahl, z.B. 3, 5, 7, ...
            sigma = final_blur_strength * 0.1  # Sigma proportional zur Stärke
            # Gaussian Blur anwenden
            gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
            swap = gaussian_blur(swap)

        if parameters['ColorNoiseDecimalSlider'] > 0:
            noise = (torch.rand_like(swap) - 0.5) * 2 * parameters['ColorNoiseDecimalSlider']
            swap = torch.clamp(swap + noise, 0.0, 255.0)

        if parameters["BlockShiftEnableToggle"]:

            #tform_scale = parameters["BlockShiftAmountSlider"] / tform.scale# / 2
            base_quality = parameters["BlockShiftAmountSlider"]  

            # s = tform.scale = Originalgröße / 512.0
            s = tform.scale  

            s = 1.0/s - 1.0
            tform_scale = int(round(base_quality + (8 - base_quality) * s))
            tform_scale = round(tform_scale)
            tform_scale = min(8, tform_scale)
            tform_scale = max(1, tform_scale)

            swap2 = self.apply_block_shift_gpu(swap, tform_scale, parameters["BlockShiftMaxAmountSlider"])        
            block_shift_blend = parameters["BlockShiftBlendAmountSlider"]/100.0#*max(1,(parameters["BlockShiftAdjustAmountSlider"]*2*tform_scale2))# * tform.scale
            #block_shift_blend = min(1, block_shift_blend)
            #block_shift_blend = max(0.1, block_shift_blend)
            if control["CommandLineDebugEnableToggle"]:
                print("MPEG Blocksize: ", tform_scale, "(resize_factor: ", tform.scale, ")")

            swap = torch.add(torch.mul(swap2, block_shift_blend), torch.mul(swap, 1 - block_shift_blend))                          
            
        if parameters['JPEGCompressionEnableToggle']:
            try: 
                '''
                jpeg_q = parameters["JPEGCompressionAmountSlider"] - ((parameters["JPEGCompressionAdjustAmountSlider"]*2) * tform.scale)
                jpeg_q = round(jpeg_q)
                jpeg_q = min(100, jpeg_q) 
                jpeg_q = max(23, jpeg_q)
                '''
                jpeg_q = parameters["JPEGCompressionAmountSlider"]
                if jpeg_q != 100:
                    '''
                    jpeg_q = 100-((100-jpeg_q)*tform.scale)
                    #jpeg_q = jpeg_q / tform.scale
                    jpeg_q = round(jpeg_q)
                    jpeg_q = min(100, jpeg_q) 
                    jpeg_q = max(15, jpeg_q)
                    '''
                    base_quality = jpeg_q  

                    # s = tform.scale = Originalgröße / 512.0
                    s = tform.scale  

                    s = 1.0/s - 1.0
                    jpeg_q = int(round(base_quality + (100 - base_quality) * s))
                    jpeg_q = max(1, min(100, jpeg_q))  # Clamp auf [1,100]
                    
                    if control["CommandLineDebugEnableToggle"]:
                        print("JPEG Quality: ", jpeg_q, " (resize_factor: ", tform.scale, ")")

                    swap2 = faceutil.jpegBlur(swap, jpeg_q)
                    blend = parameters['JPEGCompressionBlendSlider']/100
                    swap = torch.add(torch.mul(swap2, blend), torch.mul(swap, 1 - blend))                          
                    
            except:
                pass

        if parameters['AnalyseImageEnableToggle']:
            image_analyse_swap = self.analyze_image(swap)
            print("original:", image_analyse)
            print("    swap: ", image_analyse_swap)

        # Add blur to swap_mask results
        gauss = transforms.GaussianBlur(parameters['OverallMaskBlendAmountSlider'] * 2 + 1, (parameters['OverallMaskBlendAmountSlider'] + 1) * 0.2)
        swap_mask = gauss(swap_mask)

        # Combine border and swap mask, scale, and apply to swap
        swap_mask = torch.mul(swap_mask, border_mask)
        swap_mask = t512_mask(swap_mask)
        '''
        if parameters['SecondSwapEnableToggle']:
            if (s_e is not None and len(s_e) > 0) or (swapper_model2 == 'DeepFaceLive (DFM)' and dfm_model):

                bla, dfm_model, bli, blub = self.get_affined_face_dim_and_swapping_latents(original_faces, swapper_model2, dfm_model, s_e, t_e, parameters, tform)

                # Optional Scaling # change the transform matrix scaling from center
                #if parameters['FaceAdjEnableToggle']:
                #    input_face_affined = v2.functional.affine(input_face_affined, 0, (0, 0), 1 + parameters['FaceScaleAmountSlider'] / 100, 0, center=(dim*128/2, dim*128/2), interpolation=v2.InterpolationMode.BILINEAR)

                itex = 1
                #if parameters['StrengthEnableToggle']:
                #    itex = ceil(parameters['StrengthAmountSlider'] / 100.)

                # Create empty output image and preprocess it for swapping
                #output_size = int(128 * dim)
                output = torch.zeros((output_size, output_size, 3), dtype=torch.float32, device=self.models_processor.device)
                input_face_affined = swap.permute(1, 2, 0)
                original_face_512_second = original_face_512.clone()
                #original_face_512_second = original_face_512_second.permute(1, 2, 0)
                input_face_affined = torch.div(input_face_affined, 255.0)

                swap, prev_face = self.get_swapped_and_prev_face(output, input_face_affined, original_face_512_second, latent, itex, dim, swapper_model2, dfm_model, parameters)            
        '''
        swap = torch.mul(swap, swap_mask)          

        # For face comparing
        original_face_512_clone = None
        if self.is_view_face_compare:
            original_face_512_clone = original_face_512.clone()
            original_face_512_clone = original_face_512_clone.type(torch.uint8)
            original_face_512_clone = original_face_512_clone.permute(1, 2, 0)
        swap_mask_clone = None
        # Uninvert and create image from swap mask
        
        if self.is_view_face_mask:
            mask_show_type = parameters['MaskShowSelection']            
            if mask_show_type == 'swap_mask':
                if parameters['FaceEditorEnableToggle'] and self.main_window.editFacesButton.isChecked():
                    swap_mask_clone = editor_mask.clone()
                else:
                    swap_mask_clone = swap_mask.clone()
            #elif mask_show_type == 'calc_mask':
            #    swap_mask_clone = calc_mask.clone()                        
            elif mask_show_type == 'diff':
                swap_mask_clone = diff_mask.clone()                        
            elif mask_show_type == 'texture':
                swap_mask_clone = texture_mask_view.clone()                        
            #elif mask_show_type == 'mask_autocolor':
            #    swap_mask_clone = mask_autocolor.clone()                     
            #elif mask_show_type == 'restore':
            #    swap_mask_clone = swap_mask_test.clone()
            #swap_mask_clone = blubmask.clone()
            swap_mask_clone = torch.sub(1, swap_mask_clone)
            swap_mask_clone = torch.cat((swap_mask_clone,swap_mask_clone,swap_mask_clone),0)
            swap_mask_clone = swap_mask_clone.permute(1, 2, 0)
            swap_mask_clone = torch.mul(swap_mask_clone, 255.).type(torch.uint8)

        # Calculate the area to be mergerd back to the original frame
        IM512 = tform.inverse.params[0:2, :]
        corners = np.array([[0,0], [0,511], [511, 0], [511, 511]])

        x = (IM512[0][0]*corners[:,0] + IM512[0][1]*corners[:,1] + IM512[0][2])
        y = (IM512[1][0]*corners[:,0] + IM512[1][1]*corners[:,1] + IM512[1][2])

        left = floor(np.min(x))
        if left<0:
            left=0
        top = floor(np.min(y))
        if top<0:
            top=0
        right = ceil(np.max(x))
        if right>img.shape[2]:
            right=img.shape[2]
        bottom = ceil(np.max(y))
        if bottom>img.shape[1]:
            bottom=img.shape[1]

        # Untransform the swap
        swap = v2.functional.pad(swap, (0,0,img.shape[2]-512, img.shape[1]-512))
        swap = v2.functional.affine(swap, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0, interpolation=interpolation_Untransform, center = (0,0) )
        swap = swap[0:3, top:bottom, left:right]

        # Untransform the swap mask
        swap_mask = v2.functional.pad(swap_mask, (0,0,img.shape[2]-512, img.shape[1]-512))
        swap_mask = v2.functional.affine(swap_mask, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )
        swap_mask = swap_mask[0:1, top:bottom, left:right]
        swap_mask_minus = swap_mask.clone()
        swap_mask_minus = torch.sub(1, swap_mask)

        # Apply the mask to the original image areas
        img_crop = img[0:3, top:bottom, left:right]
        img_crop = torch.mul(swap_mask_minus,img_crop)
        
        #Add the cropped areas and place them back into the original image
        swap = torch.add(swap, img_crop)
        swap = swap.type(torch.uint8)
        swap = swap.clamp(0, 255)

        img[0:3, top:bottom, left:right] = swap

        return img, original_face_512_clone, swap_mask_clone

    def enhance_core(self, img, control):
        enhancer_type = control['FrameEnhancerTypeSelection']

        match enhancer_type:
            case 'RealEsrgan-x2-Plus' | 'RealEsrgan-x4-Plus' | 'BSRGan-x2' | 'BSRGan-x4' | 'UltraSharp-x4' | 'UltraMix-x4' | 'RealEsr-General-x4v3':
                tile_size = 512

                if enhancer_type == 'RealEsrgan-x2-Plus' or enhancer_type == 'BSRGan-x2':
                    scale = 2
                else:
                    scale = 4

                image = img.type(torch.float32)
                if torch.max(image) > 256:  # 16-bit image
                    max_range = 65535
                else:
                    max_range = 255

                image = torch.div(image, max_range)
                image = torch.unsqueeze(image, 0).contiguous()

                image = self.models_processor.run_enhance_frame_tile_process(image, enhancer_type, tile_size=tile_size, scale=scale)

                image = torch.squeeze(image)
                image = torch.clamp(image, 0, 1)
                image = torch.mul(image, max_range)

                # Blend
                alpha = float(control["FrameEnhancerBlendSlider"])/100.0

                t_scale = v2.Resize((img.shape[1] * scale, img.shape[2] * scale), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                img = t_scale(img)
                img = torch.add(torch.mul(image, alpha), torch.mul(img, 1-alpha))
                if max_range == 255:
                    img = img.type(torch.uint8)
                else:
                    img = img.type(torch.uint16)

            case 'DeOldify-Artistic' | 'DeOldify-Stable' | 'DeOldify-Video':
                render_factor = 384 # 12 * 32 | highest quality = 20 * 32 == 640

                _, h, w = img.shape
                t_resize_i = v2.Resize((render_factor, render_factor), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                image = t_resize_i(img)

                image = image.type(torch.float32)
                image = torch.unsqueeze(image, 0).contiguous()

                output = torch.empty((image.shape), dtype=torch.float32, device=self.models_processor.device).contiguous()

                match enhancer_type:
                    case 'DeOldify-Artistic':
                        self.models_processor.run_deoldify_artistic(image, output)
                    case 'DeOldify-Stable':
                        self.models_processor.run_deoldify_stable(image, output)
                    case 'DeOldify-Video':
                        self.models_processor.run_deoldify_video(image, output)

                output = torch.squeeze(output)
                t_resize_o = v2.Resize((h, w), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                output = t_resize_o(output)

                output = faceutil.rgb_to_yuv(output, True)
                # do a black and white transform first to get better luminance values
                hires = faceutil.rgb_to_yuv(img, True)

                hires[1:3, :, :] = output[1:3, :, :]
                hires = faceutil.yuv_to_rgb(hires, True)

                # Blend
                alpha = float(control["FrameEnhancerBlendSlider"]) / 100.0
                img = torch.add(torch.mul(hires, alpha), torch.mul(img, 1-alpha))

                img = img.type(torch.uint8)

            case 'DDColor-Artistic' | 'DDColor':
                render_factor = 384 # 12 * 32 | highest quality = 20 * 32 == 640

                # Converti RGB a LAB
                #'''
                #orig_l = img.permute(1, 2, 0).cpu().numpy()
                #orig_l = cv2.cvtColor(orig_l, cv2.COLOR_RGB2Lab)
                #orig_l = torch.from_numpy(orig_l).to(self.models_processor.device)
                #orig_l = orig_l.permute(2, 0, 1)
                #'''
                orig_l = faceutil.rgb_to_lab(img, True)

                orig_l = orig_l[0:1, :, :]  # (1, h, w)

                # Resize per il modello
                t_resize_i = v2.Resize((render_factor, render_factor), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                image = t_resize_i(img)

                # Converti RGB in LAB
                #'''
                #img_l = image.permute(1, 2, 0).cpu().numpy()
                #img_l = cv2.cvtColor(img_l, cv2.COLOR_RGB2Lab)
                #img_l = torch.from_numpy(img_l).to(self.models_processor.device)
                #img_l = img_l.permute(2, 0, 1)
                #'''
                img_l = faceutil.rgb_to_lab(image, True)

                img_l = img_l[0:1, :, :]  # (1, render_factor, render_factor)
                img_gray_lab = torch.cat((img_l, torch.zeros_like(img_l), torch.zeros_like(img_l)), dim=0)  # (3, render_factor, render_factor)

                # Converti LAB in RGB
                #'''
                #img_gray_lab = img_gray_lab.permute(1, 2, 0).cpu().numpy()
                #img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
                #img_gray_rgb = torch.from_numpy(img_gray_rgb).to(self.models_processor.device)
                #img_gray_rgb = img_gray_rgb.permute(2, 0, 1)
                #'''
                img_gray_rgb = faceutil.lab_to_rgb(img_gray_lab)

                tensor_gray_rgb = torch.unsqueeze(img_gray_rgb.type(torch.float32), 0).contiguous()

                # Prepara il tensore per il modello
                output_ab = torch.empty((1, 2, render_factor, render_factor), dtype=torch.float32, device=self.models_processor.device)

                # Esegui il modello
                match enhancer_type:
                    case 'DDColor-Artistic':
                        self.models_processor.run_ddcolor_artistic(tensor_gray_rgb, output_ab)
                    case 'DDColor':
                        self.models_processor.run_ddcolor(tensor_gray_rgb, output_ab)

                output_ab = output_ab.squeeze(0)  # (2, render_factor, render_factor)

                t_resize_o = v2.Resize((img.size(1), img.size(2)), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                output_lab_resize = t_resize_o(output_ab)

                # Combina il canale L originale con il risultato del modello
                output_lab = torch.cat((orig_l, output_lab_resize), dim=0)  # (3, original_H, original_W)

                # Convert LAB to RGB
                #'''
                #output_rgb = output_lab.permute(1, 2, 0).cpu().numpy()
                #output_rgb = cv2.cvtColor(output_rgb, cv2.COLOR_Lab2RGB)
                #output_rgb = torch.from_numpy(output_rgb).to(self.models_processor.device)
                #output_rgb = output_rgb.permute(2, 0, 1)
                #'''
                output_rgb = faceutil.lab_to_rgb(output_lab, True)  # (3, original_H, original_W)

                # Miscela le immagini
                alpha = float(control["FrameEnhancerBlendSlider"]) / 100.0
                blended_img = torch.add(torch.mul(output_rgb, alpha), torch.mul(img, 1 - alpha))

                # Converti in uint8
                img = blended_img.type(torch.uint8)

        return img

    def apply_face_expression_restorer(self, driving, target, parameters):
        """ Apply face expression restorer from driving to target.

        Args:
        driving (torch.Tensor: uint8): Driving image tensor (C x H x W)
        target (torch.Tensor: float32): Target image tensor (C x H x W)
        parameters (dict).
        
        Returns:
        torch.Tensor (uint8 -> float32): Transformed image (C x H x W)
        """
        #t256 = v2.Resize((256, 256), interpolation=interpolation_method_affine, antialias=antialias_method)

        #cv2.imwrite("driving.png", cv2.cvtColor(driving.permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR))
        _, driving_lmk_crop, _ = self.models_processor.run_detect_landmark(driving, bbox=np.array([0, 0, 512, 512]), det_kpss=[], detect_mode='203', score=0.5, from_points=False)
        driving_face_512 = driving.clone()
        #cv2.imshow("driving", cv2.cvtColor(driving_face_512.permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        driving_face_256 = t256_face(driving_face_512)

        # Making motion templates: driving_template_dct
        #print("kps: ", kps)
        c_d_eyes_lst = faceutil.calc_eye_close_ratio(driving_lmk_crop[None]) #c_d_eyes_lst
        c_d_lip_lst = faceutil.calc_lip_close_ratio(driving_lmk_crop[None]) #c_d_lip_lst
        x_d_i_info = self.models_processor.lp_motion_extractor(driving_face_256, 'Human-Face')
        R_d_i = faceutil.get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])
        ##
        
        # R_d_0, x_d_0_info = None, None
        driving_multiplier=parameters['FaceExpressionFriendlyFactorDecimalSlider'] # 1.0 # be used only when driving_option is "expression-friendly"
        animation_region = parameters['FaceExpressionAnimationRegionSelection'] # 'all' # lips, eyes, pose, exp

        flag_normalize_lip = parameters['FaceExpressionNormalizeLipsEnableToggle'] # True #inf_cfg.flag_normalize_lip  # not overwrite
        lip_normalize_threshold = parameters['FaceExpressionNormalizeLipsThresholdDecimalSlider'] # 0.03 # threshold for flag_normalize_lip
        flag_eye_retargeting = parameters['FaceExpressionRetargetingEyesEnableToggle'] # False #inf_cfg.flag_eye_retargeting
        eye_retargeting_multiplier = parameters['FaceExpressionRetargetingEyesMultiplierDecimalSlider']  # 1.00
        flag_lip_retargeting = parameters['FaceExpressionRetargetingLipsEnableToggle'] # False #inf_cfg.flag_lip_retargeting
        lip_retargeting_multiplier = parameters['FaceExpressionRetargetingLipsMultiplierDecimalSlider'] # 1.00
        
        # fix:
        if animation_region == 'all':
            animation_region = 'eyes,lips'

        flag_relative_motion = True #inf_cfg.flag_relative_motion
        flag_stitching = True #inf_cfg.flag_stitching
        flag_pasteback = True #inf_cfg.flag_pasteback
        flag_do_crop = True #inf_cfg.flag_do_crop
        
        lip_delta_before_animation, eye_delta_before_animation = None, None

        target = torch.clamp(target, 0, 255).type(torch.uint8)
        #cv2.imwrite("target.png", cv2.cvtColor(target.permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR))
        _, source_lmk, _ = self.models_processor.run_detect_landmark(target, bbox=np.array([0, 0, 512, 512]), det_kpss=[], detect_mode='203', score=0.5, from_points=False)
        target_face_512, M_o2c, M_c2o = faceutil.warp_face_by_face_landmark_x(target, source_lmk, dsize=512, scale=parameters['FaceExpressionCropScaleDecimalSlider'], vy_ratio=parameters['FaceExpressionVYRatioDecimalSlider'], interpolation=interpolation_expression_faceeditor_back)
        #cv2.imshow("target", cv2.cvtColor(target_face_512.permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        target_face_256 = t256_face(target_face_512)

        x_s_info = self.models_processor.lp_motion_extractor(target_face_256, 'Human-Face')
        x_c_s = x_s_info['kp']
        R_s = faceutil.get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.models_processor.lp_appearance_feature_extractor(target_face_256, 'Human-Face')
        x_s = faceutil.transform_keypoint(x_s_info)

        # let lip-open scalar to be 0 at first
        if flag_normalize_lip and flag_relative_motion and source_lmk is not None:
            c_d_lip_before_animation = [0.]
            combined_lip_ratio_tensor_before_animation = faceutil.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk, device=self.models_processor.device)
            if combined_lip_ratio_tensor_before_animation[0][0] >= lip_normalize_threshold:
                lip_delta_before_animation = self.models_processor.lp_retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)

        #R_d_0 = R_d_i.clone()
        #x_d_0_info = x_d_i_info.copy()

        delta_new = x_s_info['exp'].clone()
        if flag_relative_motion:
            if animation_region == "all" or animation_region == "pose":
                #R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
                R_new = (R_d_i @ R_d_i.permute(0, 2, 1)) @ R_s
            else:
                R_new = R_s
            if animation_region == "all" or animation_region == "exp":
                delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(self.models_processor.lp_lip_array).to(dtype=torch.float32, device=self.models_processor.device))
            else:
                if "lips" in animation_region:
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        delta_new[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(self.models_processor.lp_lip_array).to(dtype=torch.float32, device=self.models_processor.device)))[:, lip_idx, :]

                if "eyes" in animation_region:
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        delta_new[:, eyes_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - 0))[:, eyes_idx, :]
            '''
            elif animation_region == "lips":
                for lip_idx in [6, 12, 14, 17, 19, 20]:
                    delta_new[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(self.models_processor.lp_lip_array).to(dtype=torch.float32, device=self.models_processor.device)))[:, lip_idx, :]
            elif animation_region == "eyes":
                for eyes_idx in [11, 13, 15, 16, 18]:
                    delta_new[:, eyes_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - 0))[:, eyes_idx, :]
            '''
            if animation_region == "all":
                #scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
                scale_new = x_s_info['scale']
            else:
                scale_new = x_s_info['scale']
            if animation_region == "all" or animation_region == "pose":
                #t_new = x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
                t_new = x_s_info['t']
            else:
                t_new = x_s_info['t']
        else:
            if animation_region == "all" or animation_region == "pose":
                R_new = R_d_i
            else:
                R_new = R_s
            if animation_region == "all" or animation_region == "exp":
                for idx in [1,2,6,11,12,13,14,15,16,17,18,19,20]:
                    delta_new[:, idx, :] = x_d_i_info['exp'][:, idx, :]
                delta_new[:, 3:5, 1] = x_d_i_info['exp'][:, 3:5, 1]
                delta_new[:, 5, 2] = x_d_i_info['exp'][:, 5, 2]
                delta_new[:, 8, 2] = x_d_i_info['exp'][:, 8, 2]
                delta_new[:, 9, 1:] = x_d_i_info['exp'][:, 9, 1:]
            else:
                if "lips" in animation_region:
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        delta_new[:, lip_idx, :] = x_d_i_info['exp'][:, lip_idx, :]

                if "eyes" in animation_region:
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        delta_new[:, eyes_idx, :] = x_d_i_info['exp'][:, eyes_idx, :]
            '''
            elif animation_region == "lips":
                for lip_idx in [6, 12, 14, 17, 19, 20]:
                    delta_new[:, lip_idx, :] = x_d_i_info['exp'][:, lip_idx, :]
            elif animation_region == "eyes":
                for eyes_idx in [11, 13, 15, 16, 18]:
                    delta_new[:, eyes_idx, :] = x_d_i_info['exp'][:, eyes_idx, :]
            '''
            scale_new = x_s_info['scale']
            if animation_region == "all" or animation_region == "pose":
                t_new = x_d_i_info['t']
            else:
                t_new = x_s_info['t']

        t_new[..., 2].fill_(0)  # zero tz
        x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new
        
        if not flag_stitching and not flag_eye_retargeting and not flag_lip_retargeting:
            # without stitching or retargeting
            if flag_normalize_lip and lip_delta_before_animation is not None:
                x_d_i_new += lip_delta_before_animation

        elif flag_stitching and not flag_eye_retargeting and not flag_lip_retargeting:
            # with stitching and without retargeting
            if flag_normalize_lip and lip_delta_before_animation is not None:
                x_d_i_new = self.models_processor.lp_stitching(x_s, x_d_i_new, parameters["FaceEditorTypeSelection"]) + lip_delta_before_animation
            else:
                x_d_i_new = self.models_processor.lp_stitching(x_s, x_d_i_new, parameters["FaceEditorTypeSelection"])

        else:
            eyes_delta, lip_delta = None, None
            if flag_eye_retargeting and source_lmk is not None:
                c_d_eyes_i = c_d_eyes_lst
                combined_eye_ratio_tensor = faceutil.calc_combined_eye_ratio(c_d_eyes_i, source_lmk, device=self.models_processor.device)
                combined_eye_ratio_tensor = combined_eye_ratio_tensor * eye_retargeting_multiplier
                # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)                
                eyes_delta = self.models_processor.lp_retarget_eye(x_s, combined_eye_ratio_tensor, parameters["FaceEditorTypeSelection"])

            if flag_lip_retargeting and source_lmk is not None:
                c_d_lip_i = c_d_lip_lst
                combined_lip_ratio_tensor = faceutil.calc_combined_lip_ratio(c_d_lip_i, source_lmk, device=self.models_processor.device)
                combined_lip_ratio_tensor = combined_lip_ratio_tensor * lip_retargeting_multiplier
                # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                lip_delta = self.models_processor.lp_retarget_lip(x_s, combined_lip_ratio_tensor, parameters["FaceEditorTypeSelection"])

            if flag_relative_motion:  # use x_s
                x_d_i_new = x_s + \
                    (eyes_delta if eyes_delta is not None else 0) + \
                    (lip_delta if lip_delta is not None else 0)
            else:  # use x_d,i
                x_d_i_new = x_d_i_new + \
                    (eyes_delta if eyes_delta is not None else 0) + \
                    (lip_delta if lip_delta is not None else 0)

            if flag_stitching:
                x_d_i_new = self.models_processor.lp_stitching(x_s, x_d_i_new, parameters["FaceEditorTypeSelection"])

        x_d_i_new = x_s + (x_d_i_new - x_s) * driving_multiplier

        out = self.models_processor.lp_warp_decode(f_s, x_s, x_d_i_new, parameters["FaceEditorTypeSelection"])
        out = torch.squeeze(out)
        out = torch.clamp(out, 0, 1)  # Clip i valori tra 0 e 1

        # Applica la maschera
        #out = torch.mul(out, self.models_processor.lp_mask_crop)  # Applica la maschera

        if flag_pasteback and flag_do_crop and flag_stitching:
            with self.lock:
                t = trans.SimilarityTransform()
                t.params[0:2] = M_c2o
                dsize = (target.shape[1], target.shape[2])
                # pad image by image size
                out = faceutil.pad_image_by_size(out, dsize)
                out = v2.functional.affine(out, t.rotation*57.2958, translate=(t.translation[0], t.translation[1]), scale=t.scale, shear=(0.0, 0.0), interpolation=interpolation_expression_faceeditor_back, center=(0, 0))
                out = v2.functional.crop(out, 0,0, dsize[0], dsize[1]) # cols, rows

        img = out                
        img = torch.mul(img, 255.0)
        img = torch.clamp(img, 0, 255).type(torch.float32)        #cv2.imshow("output", cv2.cvtColor(out.permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        return img

    def swap_edit_face_core(self, img, kps, parameters, control, **kwargs): # img = RGB
        # Grab 512 face from image and create 256 and 128 copys
        if parameters['FaceEditorEnableToggle']:
            # Scaling Transforms
            #t256 = v2.Resize((256, 256), interpolation=interpolation_method_affine, antialias=antialias_method)

            # initial eye_ratio and lip_ratio values
            init_source_eye_ratio = 0.0
            init_source_lip_ratio = 0.0

            _, lmk_crop, _ = self.models_processor.run_detect_landmark( img, bbox=np.array([0, 0, 512, 512]), det_kpss=[], detect_mode='203', score=0.5, from_points=False)
            source_eye_ratio = faceutil.calc_eye_close_ratio(lmk_crop[None])
            source_lip_ratio = faceutil.calc_lip_close_ratio(lmk_crop[None])
            init_source_eye_ratio = round(float(source_eye_ratio.mean()), 2)
            init_source_lip_ratio = round(float(source_lip_ratio[0][0]), 2)

            # prepare_retargeting_image
            original_face_512, M_o2c, M_c2o = faceutil.warp_face_by_face_landmark_x(img, lmk_crop, dsize=512, scale=parameters["FaceEditorCropScaleDecimalSlider"], vy_ratio=parameters['FaceEditorVYRatioDecimalSlider'], interpolation=interpolation_expression_faceeditor_back)
            original_face_256 = t256_face(original_face_512)

            x_s_info = self.models_processor.lp_motion_extractor(original_face_256, parameters["FaceEditorTypeSelection"])
            x_d_info_user_pitch = x_s_info['pitch'] + parameters['HeadPitchSlider'] #input_head_pitch_variation
            x_d_info_user_yaw = x_s_info['yaw'] + parameters['HeadYawSlider'] # input_head_yaw_variation
            x_d_info_user_roll = x_s_info['roll'] + parameters['HeadRollSlider'] #input_head_roll_variation
            R_s_user = faceutil.get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
            R_d_user = faceutil.get_rotation_matrix(x_d_info_user_pitch, x_d_info_user_yaw, x_d_info_user_roll)
            f_s_user = self.models_processor.lp_appearance_feature_extractor(original_face_256, parameters["FaceEditorTypeSelection"])
            x_s_user = faceutil.transform_keypoint(x_s_info)

            #execute_image_retargeting
            mov_x = torch.tensor(parameters['XAxisMovementDecimalSlider']).to(self.models_processor.device)
            mov_y = torch.tensor(parameters['YAxisMovementDecimalSlider']).to(self.models_processor.device)
            mov_z = torch.tensor(parameters['ZAxisMovementDecimalSlider']).to(self.models_processor.device)
            eyeball_direction_x = torch.tensor(parameters['EyeGazeHorizontalDecimalSlider']).to(self.models_processor.device)
            eyeball_direction_y = torch.tensor(parameters['EyeGazeVerticalDecimalSlider']).to(self.models_processor.device)
            smile = torch.tensor(parameters['MouthSmileDecimalSlider']).to(self.models_processor.device)
            wink = torch.tensor(parameters['EyeWinkDecimalSlider']).to(self.models_processor.device)
            eyebrow = torch.tensor(parameters['EyeBrowsDirectionDecimalSlider']).to(self.models_processor.device)
            lip_variation_zero = torch.tensor(parameters['MouthPoutingDecimalSlider']).to(self.models_processor.device)
            lip_variation_one = torch.tensor(parameters['MouthPursingDecimalSlider']).to(self.models_processor.device)
            lip_variation_two = torch.tensor(parameters['MouthGrinDecimalSlider']).to(self.models_processor.device)
            lip_variation_three = torch.tensor(parameters['LipsCloseOpenSlider']).to(self.models_processor.device)

            x_c_s = x_s_info['kp']
            delta_new = x_s_info['exp']
            scale_new = x_s_info['scale']
            t_new = x_s_info['t']
            R_d_new = (R_d_user @ R_s_user.permute(0, 2, 1)) @ R_s_user

            if eyeball_direction_x != 0 or eyeball_direction_y != 0:
                delta_new = faceutil.update_delta_new_eyeball_direction(eyeball_direction_x, eyeball_direction_y, delta_new)
            if smile != 0:
                delta_new = faceutil.update_delta_new_smile(smile, delta_new)
            if wink != 0:
                delta_new = faceutil.update_delta_new_wink(wink, delta_new)
            if eyebrow != 0:
                delta_new = faceutil.update_delta_new_eyebrow(eyebrow, delta_new)
            if lip_variation_zero != 0:
                delta_new = faceutil.update_delta_new_lip_variation_zero(lip_variation_zero, delta_new)
            if lip_variation_one !=  0:
                delta_new = faceutil.update_delta_new_lip_variation_one(lip_variation_one, delta_new)
            if lip_variation_two != 0:
                delta_new = faceutil.update_delta_new_lip_variation_two(lip_variation_two, delta_new)
            if lip_variation_three != 0:
                delta_new = faceutil.update_delta_new_lip_variation_three(lip_variation_three, delta_new)
            if mov_x != 0:
                delta_new = faceutil.update_delta_new_mov_x(-mov_x, delta_new)
            if mov_y !=0 :
                delta_new = faceutil.update_delta_new_mov_y(mov_y, delta_new)

            x_d_new = mov_z * scale_new * (x_c_s @ R_d_new + delta_new) + t_new
            eyes_delta, lip_delta = None, None

            input_eye_ratio = max(min(init_source_eye_ratio + parameters['EyesOpenRatioDecimalSlider'], 0.80), 0.00)
            if input_eye_ratio != init_source_eye_ratio:
                combined_eye_ratio_tensor = faceutil.calc_combined_eye_ratio([[float(input_eye_ratio)]], lmk_crop, device=self.models_processor.device)
                eyes_delta = self.models_processor.lp_retarget_eye(x_s_user, combined_eye_ratio_tensor, parameters["FaceEditorTypeSelection"])

            input_lip_ratio = max(min(init_source_lip_ratio + parameters['LipsOpenRatioDecimalSlider'], 0.80), 0.00)
            if input_lip_ratio != init_source_lip_ratio:
                combined_lip_ratio_tensor = faceutil.calc_combined_lip_ratio([[float(input_lip_ratio)]], lmk_crop, device=self.models_processor.device)
                lip_delta = self.models_processor.lp_retarget_lip(x_s_user, combined_lip_ratio_tensor, parameters["FaceEditorTypeSelection"])

            x_d_new = x_d_new + \
                    (eyes_delta if eyes_delta is not None else 0) + \
                    (lip_delta if lip_delta is not None else 0)

            flag_stitching_retargeting_input: bool = kwargs.get('flag_stitching_retargeting_input', True)
            if flag_stitching_retargeting_input:
                x_d_new = self.models_processor.lp_stitching(x_s_user, x_d_new, parameters["FaceEditorTypeSelection"])

            out = self.models_processor.lp_warp_decode(f_s_user, x_s_user, x_d_new, parameters["FaceEditorTypeSelection"])
            out = torch.squeeze(out)
            out = torch.clamp(out, 0, 1)  # clip to 0~1

                #flag_do_crop_input_retargeting_image = kwargs.get('flag_do_crop_input_retargeting_image', False)
            #if flag_do_crop_input_retargeting_image:
            #    gauss = transforms.GaussianBlur(parameters['FaceEditorBlurAmountSlider']*2+1, (parameters['FaceEditorBlurAmountSlider']+1)*0.2)
            #    mask_crop = gauss(self.models_processor.lp_mask_crop)
            #    img = faceutil.paste_back_adv(out, M_c2o, img, mask_crop)
            #else:
            t = trans.SimilarityTransform()
            t.params[0:2] = M_c2o
            dsize = (img.shape[1], img.shape[2])
            # pad image by image size
            out = faceutil.pad_image_by_size(out, dsize)
            out = v2.functional.affine(out, t.rotation*57.2958, translate=(t.translation[0], t.translation[1]), scale=t.scale, shear=(0.0, 0.0), interpolation=interpolation_expression_faceeditor_back, center=(0, 0))
            out = v2.functional.crop(out, 0,0, dsize[0], dsize[1]) # cols, rows

            img = out                
            img = torch.mul(img, 255.0)
            img = torch.clamp(img, 0, 255).type(torch.float32)
                                                          
        return img
        
    def swap_edit_face_core_makeup(self, img, kps, parameters, control, **kwargs): # img = RGB
        if parameters['FaceMakeupEnableToggle'] or parameters['HairMakeupEnableToggle'] or parameters['EyeBrowsMakeupEnableToggle'] or parameters['LipsMakeupEnableToggle'] or parameters['EyesMakeupEnableToggle']:
            _, lmk_crop, _ = self.models_processor.run_detect_landmark( img, bbox=[], det_kpss=kps, detect_mode='203', score=0.5, from_points=True)

            # prepare_retargeting_image
            original_face_512, M_o2c, M_c2o = faceutil.warp_face_by_face_landmark_x(img, lmk_crop, dsize=512, scale=parameters['FaceEditorCropScaleDecimalSlider'], vy_ratio=parameters['FaceEditorVYRatioDecimalSlider'], interpolation=interpolation_expression_faceeditor_back)

            out, mask_out = self.models_processor.apply_face_makeup(original_face_512, parameters)
            if 1:
                gauss = transforms.GaussianBlur(5*2+1, (5+1)*0.2)
                out = torch.clamp(torch.div(out, 255.0), 0, 1).type(torch.float32)
                mask_crop = gauss(self.models_processor.lp_mask_crop)
                img = faceutil.paste_back_adv(out, M_c2o, img, mask_crop)

        return img
        
    @torch.no_grad()
    def gradient_magnitude(self,
                           image: torch.Tensor,
                           mask: torch.Tensor,
                           kernel_size: int,
                           weighting_strength: float,
                           sigma: float,
                           lambd: float,
                           gamma: float,
                           psi: float,
                           theta_count: int,
                           #hoch: float,
                           # CLAHE-Params
                           clip_limit: float,
                           alpha_clahe: float,
                           grid_size: tuple[int,int],
                           # Flags
                           global_gamma: float,
                           global_contrast: float,
                          ) -> torch.Tensor:
        """
        image: Tensor [C, H, W] in [0..255]
        mask:  Tensor [C, H, W] (0/1)
        Returns: Tensor [C, H, W] – gewichtete Gabor-Magnitude
        """

        C, H, W = image.shape
        eps = 1e-6
        image = image.float() / 255.0
        mask = mask.bool()
        
        # 1) Global Gamma & Kontrast
        if global_gamma != 1.0:
            image = image.pow(global_gamma)
        if global_contrast != 1.0:
            m_gc = image.mean((1,2), keepdim=True)
            image = (image - m_gc) * global_contrast + m_gc

        # 2) CLAHE im L-Kanal (mit alpha_clahe-Blending)
        if clip_limit > 0.0:
            image = image.unsqueeze(0).clamp(0,1)     # [1,3,H,W]
            mask_b3 = mask.unsqueeze(0)               # [1,3,H,W]

            lab = kc.rgb_to_lab(image)                       # [1,3,H,W]
            L   = lab[:,0:1,:,:] / 100.0                     # [1,1,H,W]

            mb = mask_b3[:,0:1,:,:]                          # [1,1,H,W]
            area_l   = mb.sum((2,3), keepdim=True).clamp(min=1)
            mean_l = (L * mb).sum((2,3), keepdim=True) / area_l
            Lf     = torch.where(mb, L, mean_l)
            Leq    = ke.equalize_clahe(
                        Lf, clip_limit=clip_limit, grid_size=grid_size,
                        slow_and_differentiable=False
                    ).clamp(0,1)
            L_blend = alpha_clahe * Leq + (1-alpha_clahe) * L
            Lnew    = torch.where(mb, L_blend, L)

            lab_eq = torch.cat([Lnew*100.0, lab[:,1:,:,:]], dim=1)  # [1,3,H,W]
            x_eq   = kc.lab_to_rgb(lab_eq)#.clamp(0,1)
            image  = x_eq.squeeze(0)

        # 3) Gabor-Filter setup
        kernel_size = max(1, 2*kernel_size - 1) #23
        if theta_count == 10:
            theta_values = torch.tensor([math.pi/4], device=image.device)
        else:
            theta_values = torch.arange(8, device=image.device) * (math.pi/8) #torch.linspace(0, math.pi, theta_count+1, device=image.device)[:-1]
            #print("theta_values: ", theta_values)
        # 4) Einziger Gabor-Filter-Aufruf
        magnitude = self.apply_gabor_filter_torch(
            image, kernel_size,
            sigma, lambd,
            gamma, psi,
            theta_values
        )  # [C, H, W]

        # 5) Invertieren
        max_mv = magnitude.amax((1,2), keepdim=True)
        inverted = max_mv - magnitude                       # [C, H, W]

        # 6) Gewichtung
        if weighting_strength > 0:
            img_m = (image * mask)
            weighted = inverted * ((1-weighting_strength)
                                   + weighting_strength * img_m)
        else:
            weighted = inverted

        return weighted * 255  # [C, H, W]
        
    def apply_gabor_filter_torch(self, image, kernel_size, sigma, lambd, gamma, psi, theta_values):
        """
        image: Tensor [C, H, W]
        theta_values: Tensor [N]
        Rückgabe: Tensor [C, H, W]
        """
        C, H, W = image.shape
        image = image.unsqueeze(0)  # → [1, C, H, W]
        
        N = theta_values.shape[0]
        
        kernels = self.get_gabor_kernels(kernel_size, sigma, lambd, gamma, psi, theta_values, image.device)  # [N, 1, k, k]

        #responses = []
        
        # kernels: [N, 1, k, k]
        # erweitere auf alle Channels:
        weight = kernels.repeat_interleave(C, dim=0)       # → [N*C, 1, k, k]
        out = F.conv2d(
            image,              # [1, C, H, W]
            weight, 
            padding=kernel_size//2,
            groups=C                         # jede Channel-Gruppe bekommt N Filter
        )  # out: [1, N*C, H, W]
        # umformen in [N, C, H, W]:
        out = out.squeeze(0).view(N, C, H, W)
        magnitudes = out.amax(dim=0)   # oder .mean(dim=0)
        return magnitudes

    def get_gabor_kernels(self, kernel_size, sigma, lambd, gamma, psi, theta_values, device):
        """
        Rückgabe: Tensor [N, 1, k, k]
        """
        half = kernel_size // 2
        y, x = torch.meshgrid(
            torch.linspace(-half, half, kernel_size, device=device),
            torch.linspace(-half, half, kernel_size, device=device),
            indexing='ij'
        )

        kernels = []
        for theta in theta_values:
            x_theta = x * torch.cos(theta) + y * torch.sin(theta)
            y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

            gb = torch.exp(-0.5 * (x_theta**2 + (gamma**2) * y_theta**2) / sigma**2)
            gb *= torch.cos(2 * math.pi * x_theta / lambd + psi)
            kernels.append(gb)

        return torch.stack(kernels).unsqueeze(1)  # → [N, 1, k, k]

    def face_restorer_auto(self, original_face_512, swap_original, swap, alpha, adjust_sharpness, scale_factor, CommandLineDebugEnableToggle, swap_mask):
        
        #swap_mask = torch.where(swap_mask > 0.5, 1, 0)
        original_face_512_autorestore = original_face_512.clone().float()
        #original_face_512 = torch.where(swap_mask, original_face_512, 0)        
        #original_face_512 = original_face_512 * swap_mask

        swap_restorecalc = swap.clone()
        #swap = torch.where(swap_mask, swap, original_face_512)
        #swap = swap * swap_mask

        swap_original_autorestore = swap_original.clone()
        #swap_original = torch.where(swap_mask, swap_original, original_face_512)
        #swap_original = swap_original * swap_mask

        scores_original = self.sharpness_score(original_face_512)
        score_new_original = scores_original["combined"].item()*100 + adjust_sharpness/10
        alpha = 0.5
        max_iterations = 7
        alpha_min, alpha_max = 0.0, 1.0
        tolerance = 0.2
        min_alpha_change = 0.05
        iteration = 0
        prev_alpha = alpha
        iteration_blur = 0

        while iteration < max_iterations:
            swap2 = swap * alpha + swap_original * (1 - alpha)
            #swap2 = swap2 * swap_mask
            swap2_masked = swap2.clone() #torch.where(swap_mask > 0.01, swap2, original_face_512)
            scores_swap = self.sharpness_score(swap2_masked)
            score_new_swap = scores_swap["combined"].item()*100
            sharpness_diff = score_new_swap - score_new_original
            #print("Restore Blend: ", prev_alpha*100, "Iterations: ", iteration+1, "orig/swap: ", score_new_original, score_new_swap, sharpness_diff)#, tenengrad_thresh, comb_weight)
            if abs(sharpness_diff) < tolerance:
                break

            if sharpness_diff < 0:
                alpha_min = alpha
                alpha = (alpha + alpha_max) / 2
            else:
                alpha_max = alpha
                alpha = (alpha + alpha_min) / 2

            if abs(prev_alpha - alpha) < min_alpha_change:
                prev_alpha = (prev_alpha + alpha) / 2
                break

            prev_alpha = alpha
            iteration += 1

        if CommandLineDebugEnableToggle:
            print("Restore Blend: ", prev_alpha*100, "Iterations: ", iteration+1)#, tenengrad_thresh, comb_weight)

        return prev_alpha, iteration_blur
              
    def sharpness_score(
        self,
        image: torch.Tensor,
        mask: torch.Tensor = None,
        tenengrad_thresh: float = 0.05,
        comb_weight: float = 0.5
    ) -> dict:
        """
        Berechnet drei Sharpness‐Metriken auf einem RGB-Image:
          1) var_lap: Variance of Laplacian
          2) tten: Thresholded Tenengrad (Anteil starker Kanten)
          3) combined: comb_weight*var_lap + (1-comb_weight)*tten

        Args:
            image: Tensor [3, H, W], float in [0..1]
            mask:  optional Tensor [H, W] oder [1, H, W] mit 1=gültig, 0=ignorieren
            tenengrad_thresh: Schwellwert für Tenengrad (0..1)
            comb_weight: Gewicht für var_lap in der Kombi (0..1)

        Returns:
            {
              "var_lap": float Tensor,
              "ttengrad": float Tensor,
              "combined": float Tensor
            }
        """
        image = image / 255.0
        
        # 1) Graustufen [1,1,H,W]
        gray = image.mean(dim=0, keepdim=True).unsqueeze(0)

        # 2) Optional Mask auf [H,W]
        if mask is not None:
            m = mask.float()
            if m.dim() == 3:  # [1,H,W]
                m = m.squeeze(0)
        else:
            m = None
            #print("no mask")

        # Hilfs: Anzahl gültiger Pixel
        def valid_count(t):
            return m.sum().clamp(min=1.0) if m is not None else t.numel()

        # --- Variance of Laplacian ---
        lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],
                           device=image.device, dtype=torch.float32).view(1,1,3,3)
        L = F.conv2d(gray, lap, padding=1).squeeze()   # [H,W]
        L2 = L.pow(2)
        # Mask anwenden
        if m is not None:
            L  = L * m
            L2 = L2 * m
        cnt = valid_count(L2)
        mean_L2 = L2.sum() / cnt
        mean_L  = L.sum()  / cnt
        var_lap = (mean_L2 - mean_L.pow(2)).clamp(min=0.0)

        # --- Thresholded Tenengrad ---
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                               device=image.device, dtype=torch.float32).view(1,1,3,3)
        sobel_y = sobel_x.transpose(2,3)
        Gx = F.conv2d(gray, sobel_x, padding=1).squeeze()  # [H,W]
        Gy = F.conv2d(gray, sobel_y, padding=1).squeeze()
        G  = (Gx.pow(2) + Gy.pow(2)).sqrt()
        if m is not None:
            G = G * m
        total = cnt
        strong = (G > tenengrad_thresh).float().sum()
        ttengrad = strong / total

        # --- Kombinierter Score ---
        combined = comb_weight * var_lap + (1 - comb_weight) * ttengrad

        return {
            "var_lap":    var_lap,
            "ttengrad":   ttengrad,
            "combined":   combined
        }        
       
    def apply_block_shift_gpu(self, img, block_size=8, shift_max=2):
        """
        Simuliert eine Blockverschiebung wie bei schlechter MPEG-Kompression.
        GPU-optimiert ohne Schleifen.

        - img: PyTorch Tensor mit Shape (C, H, W), Wertebereich [0,255], auf GPU
        - block_size: Größe der Blöcke (z. B. 8 oder 16)
        - shift_max: Maximale Verschiebung in Pixeln für jeden Block

        Rückgabe:
        - Verzerrtes Bild als Tensor (C, H, W), bleibt auf GPU
        """

        block_size = 2 ** block_size
        C, H, W = img.shape
        img = img.float()

        # Sicherstellen, dass Höhe/Breite durch block_size teilbar sind
        H_crop = H - (H % block_size)
        W_crop = W - (W % block_size)
        img = img[:, :H_crop, :W_crop]

        # Blöcke berechnen
        H_blocks = H_crop // block_size
        W_blocks = W_crop // block_size

        # Zufällige Verschiebungen pro Block
        shift_x = torch.randint(-shift_max, shift_max + 1, (H_blocks, W_blocks), device=img.device)
        shift_y = torch.randint(-shift_max, shift_max + 1, (H_blocks, W_blocks), device=img.device)

        # Erstelle Grid für grid_sample
        base_grid = F.affine_grid(torch.eye(2, 3, device=img.device).unsqueeze(0), 
                                  [1, C, H_crop, W_crop], align_corners=False)
        
        # Skalieren, um Pixelverschiebung korrekt abzubilden
        shift_x = shift_x.float() * (2 / W_crop)
        shift_y = shift_y.float() * (2 / H_crop)

        # In Grid umwandeln (Pixel → Normalisierte Koordinaten)
        shift_x = shift_x.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
        shift_y = shift_y.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)

        # Grid anpassen
        base_grid[..., 0] += shift_x
        base_grid[..., 1] += shift_y

        # Bild verzerren
        distorted_img = F.grid_sample(img.unsqueeze(0), base_grid, mode='bilinear', padding_mode='border', align_corners=False)
        
        return distorted_img.squeeze(0).clamp(0, 255)
        

        
    def analyze_image(self, image):
        """
        Analysiert ein Bild, um Qualitätsprobleme zu erkennen (JPEG-Artefakte, Rauschen, Unschärfe, Kontrast).
        
        Args:
            image (torch.Tensor): Eingabebild als Tensor mit Shape (C, H, W), Wertebereich [0, 1].
            
        Returns:
            dict: Analyseergebnisse mit Wahrscheinlichkeiten für verschiedene Artefakte.
        """
        image = image.float() /255.0
        C, H, W = image.shape
        grayscale = torch.mean(image, dim=0, keepdim=True)  # In Graustufen umwandeln
        
        analysis = {}

        # **1️⃣ JPEG-Artefakte erkennen (Hochfrequenz-Anteile analysieren)**
        fft = torch.fft.fft2(grayscale)  # Fourier-Transformation
        high_freq_energy = torch.mean(torch.abs(fft))  # Mittlere Frequenz-Energie
        analysis["jpeg_artifacts"] = min(high_freq_energy.item() / 50, 1.0)  # Normierung

        # **2️⃣ Salt & Pepper Noise erkennen**
        median_filtered = F.avg_pool2d(grayscale, 3, stride=1, padding=1)  # Mittelwertfilter
        noise_map = torch.abs(grayscale - median_filtered)
        sp_noise = torch.mean((noise_map > 0.1).float())  # Pixel mit starkem Abweichungen
        analysis["salt_pepper_noise"] = min(sp_noise.item() * 10, 1.0)

        # **3️⃣ Speckle Noise erkennen (Varianz der Pixelwerte)**
        local_var = F.avg_pool2d(grayscale**2, 5, stride=1, padding=2) - (F.avg_pool2d(grayscale, 5, stride=1, padding=2) ** 2)
        speckle_noise = torch.mean(local_var)
        analysis["speckle_noise"] = min(speckle_noise.item() * 50, 1.0)

        # **4️⃣ Unschärfe detektieren (Kantenanalyse mit Laplace-Filter)**
        laplace_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=image.device).unsqueeze(0).unsqueeze(0)
        laplace_edges = F.conv2d(grayscale.unsqueeze(0), laplace_kernel, padding=1)
        edge_strength = torch.mean(torch.abs(laplace_edges))
        analysis["blur"] = 1.0 - min(edge_strength.item() * 5, 1.0)  # Je weniger Kanten, desto unschärfer

        # **5️⃣ Kontrastanalyse (Histogramm-Spread prüfen)**
        contrast = grayscale.std()
        analysis["low_contrast"] = 1.0 - min(contrast.item() * 10, 1.0)  # Niedrige Standardabweichung = wenig Kontrast

        return analysis