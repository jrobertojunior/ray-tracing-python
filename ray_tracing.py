import json
import numpy as np
from matplotlib import pyplot as plt

def render(v_res, h_res, square_side, dist, eye, look_at, up, bkgd_color, max_depth, objects, ca, lights):
    w = normal(eye - look_at)
    u = normal(np.cross(up, w))
    v = np.cross(w, u)
    img = np.full((v_res, h_res, 3), bkgd_color/255)
    q0 = eye - dist * w + square_side * (((v_res - 1) * v / 2) - ((h_res - 1) * u / 2))
    for i in range(v_res):
        for j in range(h_res):
            q = q0 + square_side * (j * u - i * v)
            color = cast(eye, normal((q - eye)), bkgd_color/255, objects, ca, lights, max_depth)
            color = color/max(*color, 1)
            img[i, j] = color
    return img
    
    
def cast(origin, d, color, objects, ca, lights, max_depth):
    s = trace(origin, d, objects)
    if len(s) != 0:
        t, closest = s[np.argmin(np.array(s)[:, 0])]
        p = origin + t*d
        w0 = -d
        if list(closest)[-1] == 'sphere':
          n = normal(p - closest['sphere']['center'])
        if list(closest)[-1] == 'plane':
          n = normal(closest['plane']['normal'])

        color = shade(closest, p, w0, n, ca, lights, objects)
        if max_depth > 0:
            kr = closest['kr']
            kt = closest['kt']
            if kt > 0:
              try:
                r = refract(closest,p,w0,n)
                p_ = p + 1e-5*r
                color = color + kt*cast(p_,r,color,objects,ca,lights,max_depth-1)
              except:
              	kr = 1
              	pass
            if kr > 0:
              r = reflect(w0,n)
              p_ = p + 1e-5*r
              color = color + kr*cast(p_,r,color,objects,ca,lights,max_depth-1)
    return color

def trace(origin, d, objects):
    s = []
    
    for obj in objects:
        try:
            if list(obj)[-1] == 'sphere':
                t = intersection_sphere(obj['sphere'], origin, d)
            elif list(obj)[-1] == 'plane':
                t = intersection_plane(obj['plane'], origin, d)
            s.append((t, obj))
        except:
            pass
    return s

def normal(v):
    return v / np.linalg.norm(v)

def intersection_plane(plane, origin, d):
    sample = plane['sample']
    normal = plane['normal']
    aux = np.dot(d,normal)
    a = np.linalg.norm(aux)
    if a > pow(10, -6):
        t = np.dot((sample - origin), normal) / aux
        if t <= 0:
            raise Exception("oi")
        else:
            return t
    else:
        raise Exception("oi")

def intersection_sphere(sphere, origin, d):
    center = sphere['center']
    radius = sphere['radius']
    center = np.array(center)
    L = center - origin
    tca = np.dot(L,d)
    d = np.sqrt(np.dot(L,L) - tca**2)
    if d**2 > radius**2:
        raise Exception("oi")
    else:
        thc = np.sqrt(radius**2 - d**2)
        t0 = tca - thc
        t1 = tca + thc
        if t0 > t1:
            t0, t1 = t1, t0
        if t0 <= 0:
            if t1 <= 0:
                raise Exception("oi")
            else:
                return t1
        else:
            return t0

def reflect(l, n):
    return 2*(n*np.dot(n, l)) - l

def shade(obj, p, w0, n, ca, lights, objects):
    ca = ca/255
    cd = np.array(obj['color'])/255
    ka = obj['ka']
    kd = obj['kd']
    ks = obj['ks']
    exp = obj['exp']

    cp = ka * cd * ca

    for light in lights:
        cj = np.array(light['intensity'])/255
        l = np.array(light['position'])

        lj = normal(l - p)

        rj = reflect(lj, n)

        p_ = p + lj*(1e-5)
        s = trace(p_, lj, objects)

        if len(s) > 0:
            t = s[np.argmin(np.array(s)[:, 0])][0]
        
        if len(s) == 0 or np.dot(lj, l - p_) < t:
            if np.dot(n, lj) > 0:
                cp += (kd*cd)*(np.dot(n, lj)*cj)
            if np.dot(w0, rj) > 0:
                cp += ks*(np.dot(w0, rj)**exp)*cj
    return cp

def refract(obj,p,w0,n):
    ref_n = obj['index_of_refraction']
    cosi = np.dot(n,w0)

    if cosi < 0:
        n = -n
        ref_n = 1/ref_n
        cosi = -cosi
        
    delta = 1 - ((1/(ref_n**2))*(1-cosi**2))
    if delta < 0:
        raise Exception("delta < 0")
        
    return (-1/ref_n)*w0 - (np.sqrt(delta) - (1/ref_n)*cosi)*n


def main():
    filename = 'custombolha'
    f = open(f'{filename}.json', 'r')
    data = json.load(f)
    to_np = ['eye', 'look_at', 'up', 'background_color', 'ambient_light']
    for key in to_np:
        data[key] = np.array(data[key])
        
    rendered_img = render(*data.values())
    plt.imshow(rendered_img)
    plt.show()

if __name__ == '__main__':
    main()
