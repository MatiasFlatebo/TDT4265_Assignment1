import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE
    def compute_intersection(prediction_box, gt_box):
        xmin = max(prediction_box[0], gt_box[0])
        ymin = max(prediction_box[1], gt_box[1])
        xmax = min(prediction_box[2], gt_box[2])
        ymax = min(prediction_box[3], gt_box[3])
        return max(0, xmax - xmin) * max(0, ymax - ymin) # Return area, L*B
    
    # Total area of both boxes minus the intersection to avoid double
    def compute_union(prediction_box, gt_box):
        total_area_of_prediction_box = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1])
        total_area_of_gt_box = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        return total_area_of_prediction_box + total_area_of_gt_box - compute_intersection(prediction_box, gt_box)

    intersection = compute_intersection(prediction_box, gt_box)
    union = compute_union(prediction_box, gt_box)
    iou = intersection / union

    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    return num_tp / (num_tp + num_fp) if num_tp + num_fp > 0 else 1


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    return num_tp / (num_tp + num_fn) if num_tp + num_fn > 0 else 0


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold
    correct_pred_boxes = []
    correct_gt_boxes = []
    for gt_box in gt_boxes:
        best_box, best_intersection = None, 0
        for pred_box in prediction_boxes:
            pred_iou = calculate_iou(pred_box, gt_box)
            if best_box is None or  pred_iou > best_intersection:
                if pred_iou >= iou_threshold:
                    best_box = pred_box
                    best_intersection = calculate_iou(pred_box, gt_box)
        if best_box is not None:
            correct_pred_boxes.append(best_box)
            correct_gt_boxes.append(gt_box)

    # Sort all matches on IoU in descending order
    # Find all matches with the highest IoU threshold

    return np.array(correct_pred_boxes), np.array(correct_gt_boxes)


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    correct_pred_boxes, _ = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    true_pos = correct_pred_boxes.shape[0]
    false_pos = prediction_boxes.shape[0] - true_pos
    false_neg = gt_boxes.shape[0] - true_pos
    
    return {"true_pos": true_pos, "false_pos": false_pos, "false_neg": false_neg}


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    precision, recall = 0, 0
    for i in range(len(all_gt_boxes)):
        measures_dict = calculate_individual_image_result(all_prediction_boxes[i], all_gt_boxes[i], iou_threshold)
        precision += calculate_precision(measures_dict["true_pos"], measures_dict["false_pos"], measures_dict["false_neg"])
        recall += calculate_recall(measures_dict["true_pos"], measures_dict["false_pos"], measures_dict["false_neg"])
    return precision/len(all_gt_boxes), recall/len(all_gt_boxes)



def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE
    precisions = []
    recalls = []

    for threshold in confidence_thresholds:
        filtered_prediction_boxes = []
        filtered_gt_boxes = []
        for i in range(len(all_prediction_boxes)):
            filtered_prediction_boxes.append(all_prediction_boxes[i][confidence_scores[i] >= threshold])
            filtered_gt_boxes.append(all_gt_boxes[i])
        precision, recall = calculate_precision_recall_all_images(filtered_prediction_boxes, filtered_gt_boxes, iou_threshold)
        precisions.append(precision)
        recalls.append(recall)

                
    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision-recall curve and saves the figure to precision_recall_curve.png."""
    plt.figure(figsize=(10, 8)) 
    plt.plot(recalls, precisions,
             label='Precision-Recall Curve')
    
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.title("Precision-Recall Curve", fontsize=20) 
    plt.grid() 
    
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14) 
    
    plt.xlim([0.0, 1.0]) 
    plt.ylim([0.0, 1.1]) 
    
    plt.legend(fontsize=14) 
    
    plt.savefig("precision_recall_curve_enhanced.png") 
    plt.show()


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    recall_levels = np.linspace(0, 1.0, 11)
    precisions_interpolated = np.zeros_like(recall_levels)
    
    for i, recall_level in enumerate(recall_levels):
        precisions_at_or_above_recall = precisions[recalls >= recall_level]
        if precisions_at_or_above_recall.size > 0:
            precisions_interpolated[i] = np.max(precisions_at_or_above_recall)
        else:
            precisions_interpolated[i] = 0
    average_precision = np.mean(precisions_interpolated)
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
