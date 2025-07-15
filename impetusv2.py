import pandas as pd
from IMPython.main import run_main

from shapely.geometry import Polygon
from shapely.affinity import rotate
import numpy as np

def plate_weight_calc(density,thickness,angles):
    
    total_weight=density[0]*(((thickness[0]/1000)/(np.cos(np.radians(angles[0]))))) +density[1]*(((thickness[1]/1000)/(np.cos(np.radians(angles[1])))))  #must update these values

    return total_weight

def create_box_vertices(min_point, max_point):
    """
    Create the vertices from the minimum and maximum coordinates
    """
    x_min, y_min = min_point
    x_max, y_max = max_point
    
    vertices = np.array([
        [x_min, y_min],  # 0: bottom-left
        [x_max, y_min],  # 1: bottom-right
        [x_max, y_max],  # 2: top-right
        [x_min, y_max]   # 3: top-left
    ])
    
    return vertices

def project_polygon(vertices, axis):
    """Project polygon onto an axis and return min/max scalar values."""
    projections = [np.dot(v, axis) for v in vertices]
    return min(projections), max(projections)


def overlap_on_axis(box1, box2, axis, tolerance=1e-6):
    """Check if projections of both polygons overlap on a given axis, ignoring slight edge contacts."""
    min1, max1 = project_polygon(box1, axis)
    min2, max2 = project_polygon(box2, axis)

    # If overlap is less than tolerance, treat it as no overlap
    overlap_amount = min(max1, max2) - max(min1, min2)
    
    return overlap_amount > tolerance  # Only consider it an overlap if it is significant

def get_normals(vertices):
    """Get perpendicular (normal) vectors to each edge of the polygon."""
    normals = []
    for i in range(len(vertices)):
        edge = np.subtract(vertices[(i + 1) % len(vertices)], vertices[i])
        normal = np.array([-edge[1], edge[0]])  # Perpendicular to the edge
        normal /= np.linalg.norm(normal)  # Normalize
        normals.append(normal)
    return normals

def polygons_overlap(box1, box2):
    """Check if two 2D convex polygons overlap using SAT."""
    normals = get_normals(box1) + get_normals(box2)  # All possible separating axes
    for axis in normals:
        if not overlap_on_axis(box1, box2, axis):  
            return False  # Found a separating axis → no overlap
    return True  # No separating axis found → overlap


def rotate_vertices(vertices, theta):
    """Rotate the vertices of a shape around its centroid"""
    # Compute the centroid
    centroid = polygon_centroid(vertices)
    
    # Translate points to origin
    translated_vertices = vertices - centroid
    
    # Rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Rotate vertices
    rotated_vertices = np.dot(translated_vertices, R.T)
    
    # Translate back
    final_vertices = rotated_vertices + centroid
    
    return final_vertices


def create_box_vertices(min_point, max_point):
    """
    Create the vertices from the minimum and maximum coordinates
    """
    x_min, y_min = min_point
    x_max, y_max = max_point
    
    vertices = np.array([
        [x_min, y_min],  # 0: bottom-left
        [x_max, y_min],  # 1: bottom-right
        [x_max, y_max],  # 2: top-right
        [x_min, y_max]   # 3: top-left
    ])
    
    return vertices

def polygon_centroid(vertices):
    """Calculate the centroid of a 2D polygon given its vertices."""
    n = len(vertices)
    area = 0
    Cx = 0
    Cy = 0

    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]  # Next vertex (wrap around)
        cross = x0 * y1 - x1 * y0
        area += cross
        Cx += (x0 + x1) * cross
        Cy += (y0 + y1) * cross

    area *= 0.5
    # Handle case where area might be negative
    area = abs(area)
    
    if abs(area) < 1e-10:  # Avoid division by near-zero
        return np.mean(vertices, axis=0)
    
    Cx /= (6 * area)
    Cy /= (6 * area)

    return np.array([Cx, Cy])

def plate_vertices(thick, length, angle):
    """Calculate the vertices of a plate centreed at the origin"""
    # Create plate centreed at origin
    min_point = [-thick/2, -length/2]
    max_point = [thick/2, length/2]    

    # Create vertices
    original_vertices = create_box_vertices(min_point, max_point)
    
    # Rotate the box to the required angle
    final_vertices = rotate_vertices(original_vertices, angle)

    return final_vertices

def calculate_plate_location(plate1_vertices, plate2_vertices, angle1, angle2, thickness1, thickness2,gap):
    "Calculate the shift so the location is from the back surface of one plate to the front of the other"
     # Calculate the centroids of both plates
    plate1_centre = polygon_centroid(plate1_vertices)
    plate2_centre = polygon_centroid(plate2_vertices)
    
    # Calculate the x-projection of the thickness for both plates
    # This accounts for how much x-distance the plates occupy based on their angles
    x_proj1 = thickness1/2 /np.cos(angle1) 
    x_proj2 = thickness2/2 / np.cos(angle2)
    
    # Calculate new x position for plate2
    # plate1's centre + half of plate1's x-projection + gap + half of plate2's x-projection
    new_x = plate1_centre[0] + x_proj1 + gap + x_proj2
    
    # Keep the same y-coordinate for horizontal alignment
    # If you want to position them at different heights, modify this
    new_y = plate2_centre[1]
    
    # Calculate translation vector needed
    translation = np.array([new_x - plate2_centre[0], new_y - plate2_centre[1]])
    
    # Apply translation to all vertices
    return plate2_vertices + translation

def compute_safe_rotation_bounds(plate1, plate2, angle1, thickness, gap, plate_length,max_angle_deg=60, angle_step_deg=15):
    """
    Find the upper and lower safe rotation angles for plate2 to avoid overlapping with plate1.
    """
    angles_deg = np.arange(-max_angle_deg, max_angle_deg + angle_step_deg, angle_step_deg)
    safe_angles = []

    #check to see if the gap is 0 or within 1e-2
    if abs(gap[1])<0.1:
        #the plate pretty much touching, must take the angle of the front plate
        safe_angles.append(angle1)

    else:

        for angle in np.deg2rad(angles_deg):
            plate2 = plate_vertices(thickness[1], plate_length, angle)
            plate2=calculate_plate_location(plate1, plate2, angle1, angle, thickness[0], thickness[1],gap[1])  #this is needed for every case that isnt

            overlap=polygons_overlap(plate1, plate2)
            if overlap ==False:
                safe_angles.append(angle)

    if not safe_angles:
        safe_angles.append(angle1)

    lower_angle2 = min(safe_angles)
    upper_angle2 = max(safe_angles)

    return lower_angle2, upper_angle2


def runner(h):
    #--------------Extra input data----------------------
    strelabeldict={
        1200e6: "RHA(MARS380)",
        1450e6: "HHA(ARMOX500)",
        393e6: "AA6070-T6",
        1290e6: "Ti-6Al-4V"     
        }
    zdim=0.08 # zdirection

    ipath="/vast/dylana/BO-Muse-master-update-vpthon/src/cylinder_plate.k"
    bo = "/vast/dylana/BO-Muse-master-update-vpthon/src/"
    opath="/vast/dylana/BO-Muse-master-update-vpthon/src"

    #--------------------------------------------------
    #specify the maximum and minium allowable design variables based on the material. This requires first finding the plate materials
    #min_max_df = pd.read_csv('\home\dylana\github\BO-Muse\python\min_max_values.txt', sep='\t', index_col=0)
    #mins = min_max_df['Min'].to_numpy() 
    #maxs = min_max_df['Max'].to_numpy()    

    #just adding the min_max_df values as a list
    maxs=[80.0,60.0,80.0,60.0,100.0,7850.0,1450000000.0,0.14,7850.0,1450000000.0,0.14,863.9999999999999]
    mins=[5.0,-60.0,5.0,-60.0,0.0,2700.0,393000000.0,0.07,2700.0,393000000.0,0.07,78.4]

    #Accept variables as arguments from the BO
    thick1=h[0]*(maxs[0]-mins[0]) + mins[0]
    thick2=h[2]*(maxs[2]-mins[2]) + mins[2]
    thick=[thick1,thick2]

    angle1=h[1]*(maxs[1]-mins[1]) + mins[1]    
    angle2=h[3]*(maxs[3]-mins[3]) + mins[3]
    angles=[angle1,angle2] #angles array

    gap1=float(0)
    gap2=h[4]*(maxs[4]-mins[4]) + mins[4]   

    gap_list=[gap1,gap2] #plate gaps

    stren1=h[6]*(maxs[6]-mins[6]) + mins[6]   
    stren2=h[9]*(maxs[9]-mins[9]) + mins[9]   

    #find the material label based on inputted material data
    labels_total=[strelabeldict.get(stren1),strelabeldict.get(stren2),"Tungstenalloy"]


    pval=run_main(ipath,bo,opath,labels_total,thick,angles,gap_list,zdim) #run IMPETUS and calculate the pval
    pval=-1*(pval)/(240)

    return (None, 0, [pval])

def runner_(h):
    "This allows for variable plates and continuous data"


    #--------------Extra input data----------------------
    strelabeldict={
        1200e6: "RHA(MARS380)",
        1450e6: "HHA(ARMOX500)",
        393e6: "AA6070-T6",
        1290e6: "Ti-6Al-4V"     
        }

    matthickness={ 
    "RHA(MARS380)":          [5, 10, 15, 20],
    "HHA(ARMOX500)":         [5, 10, 15, 20],
    "AA6070-T6":             [20,40, 60, 80],
    "Ti-6Al-4V":             [10,20,30,40],
    "UHMWPE(Dyneema)":       [20,40,60,80],
    "Siliconcarbide":        [10, 25, 40],
    "AluminaCeramic":        [10,25,40]
    }

    zdim=0.08 # zdirection

    #specify the maximum and minium allowable design variables based on the material. This requires first finding the plate materials
    #min_max_df = pd.read_csv('\home\dylana\github\BO-Muse\python\min_max_values.txt', sep='\t', index_col=0)
    #mins = min_max_df['Min'].to_numpy() 
    #maxs = min_max_df['Max'].to_numpy()    

    #just adding the min_max_df values as a list
    maxs=[80.0,60.0,80.0,60.0,100.0,7850.0,1450000000.0,0.14,7850.0,1450000000.0,0.14,863.9999999999999]
    mins=[5.0,-60.0,5.0,-60.0,0.0,2700.0,393000000.0,0.07,2700.0,393000000.0,0.07,78.4]    

    stren1=h[6]*(maxs[6]-mins[6]) + mins[6]   
    stren2=h[9]*(maxs[9]-mins[9]) + mins[9]   

    #find the material label based on inputted material data
    labels_total=[strelabeldict.get(stren1),strelabeldict.get(stren2),"Tungstenalloy"]

    #since thickness is material dependent, we first need to check the plate material before find the range of possible thickness
    thick_range1=[min(matthickness.get(labels_total[0])),max(matthickness.get(labels_total[0]))]
    thick_range2=[min(matthickness.get(labels_total[1])),max(matthickness.get(labels_total[1]))]

    angle_range = [-60,60]
    gap_range   = [0,100]
    plate_length = 200

    ipath="/vast/dylana/BO-Muse-master-update-vpthon/src/cylinder_plate.k"
    bo = "/vast/dylana/BO-Muse-master-update-vpthon/src/"
    opath="/vast/dylana/BO-Muse-master-update-vpthon/src"

    
    # ipath=r"C:\Users\dylana\OneDrive - Deakin University\IMPETUS-Python\data_input\cylinder_plate.k" #input file path
    # bo = r"C:\Users\dylana\OneDrive - Deakin University\IMPETUS-Python\data_input" # path where all input file information is stored
    # opath=r"C:\Users\dylana\OneDrive - Deakin University\IMPETUS-Python\data_output" #Impetus output location

    #--------------------------------------------------
    #Extract the thicknesses
    thick1=h[0]*(thick_range1[1]-thick_range1[0]) + thick_range1[0]
    thick2=h[2]*(thick_range2[1]-thick_range2[0]) + thick_range2[0]
    thick=[thick1,thick2]

    #Extract angle since this is based on the min and max allowable angles
    angle1=h[1]*(angle_range[1]-angle_range[0]) + angle_range[0]   
    # Convert degrees to radians
    angle1rad = np.radians(angle1) 

    #Extract gaps
    gap1=float(0)
    gap2=h[4]*(gap_range[1]-gap_range[0]) + gap_range[0]   
    gap_list=[gap1,gap2] #plate gaps

    # Create plates
    plate1 = plate_vertices(thick1, plate_length, angle1rad)
    plate2 = plate_vertices(thick2, plate_length, 0) # start with with vertices without orientation

    #Plate 2 needs to be at a distance measured from the back surface of one plate to the front surface of the other
    plate2=calculate_plate_location(plate1, plate2, angle1rad, 0, thick1, thick2,gap2)  #this is needed for every case that isnt

    #To calculate the safe orientations the following is used:
    lower, upper = compute_safe_rotation_bounds(plate1, plate2,angle1rad,thick,gap_list,plate_length)

    #Using the upper and lower bound angles of the second plate, calcualte the orientaiton
    angle2 = (upper - lower) * h[3] + lower
    angles = [angle1, np.degrees(angle2)]


    pval=run_main(ipath,bo,opath,labels_total,thick,angles,gap_list,zdim) #run IMPETUS and calculate the pval
    pval=-1*(pval)/(240)

    density1=h[5]*(maxs[5]-mins[5]) + mins[5]   
    density2=h[8]*(maxs[8]-mins[8]) + mins[8]   

    density=[density1,density2] #density of material plates


    tw=plate_weight_calc(density,thick,angles) #calulcate the total weight
    tw_norm=(tw-mins[11])/(maxs[11]-mins[11])

    with open("output.txt", "w") as f:
        f.write(str(pval))

    return (float(tw_norm), float(0), [float(pval)])
