import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math
import random
from time import time
from OpenGL.arrays import vbo

G_OBJ_SPHERE = None
G_OBJ_CYLINDER = None
G_OBJ_DISK = None
G_OBJ_GRID = None
G_OBJ_TREE = None
G_OBJ_WALLS = None

def lerp(start, end, t):
    return start + t * (end - start)

def smooth_damp(current, target, current_velocity, smooth_time, max_speed, delta_time):
    smooth_time = max(0.0001, smooth_time)
    omega = 2.0 / smooth_time
    x = omega * delta_time
    exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)

    change = current - target
    target_copy = target
    max_change = max_speed * smooth_time

    change = np.clip(change, -max_change, max_change)
    target = current - change

    temp = (current_velocity + omega * change) * delta_time
    current_velocity = (current_velocity - omega * temp) * exp
    output = target + (change + temp) * exp

    if ((target_copy - current) * (output - target_copy)) > 0:
        output = target_copy
        current_velocity = 0

    return output, current_velocity

def draw_text(x, y, text):
    glRasterPos2f(x, y)
    for character in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(character))

def init_primitives():
    global G_OBJ_SPHERE, G_OBJ_CYLINDER, G_OBJ_DISK, G_OBJ_GRID, G_OBJ_TREE, G_OBJ_WALLS

    base = glGenLists(6)
    G_OBJ_SPHERE = base
    G_OBJ_CYLINDER = base + 1
    G_OBJ_DISK = base + 2
    G_OBJ_GRID = base + 3
    G_OBJ_TREE = base + 4
    G_OBJ_WALLS = base + 5

    glNewList(G_OBJ_SPHERE, GL_COMPILE)
    quadric = gluNewQuadric()
    gluQuadricNormals(quadric, GLU_SMOOTH)
    gluSphere(quadric, 1.0, 16, 16)
    glEndList()

    glNewList(G_OBJ_CYLINDER, GL_COMPILE)
    quadric = gluNewQuadric()
    gluQuadricNormals(quadric, GLU_SMOOTH)
    gluCylinder(quadric, 0.1, 0.1, 1.0, 8, 1)
    glEndList()

    glNewList(G_OBJ_DISK, GL_COMPILE)
    quadric = gluNewQuadric()
    gluQuadricNormals(quadric, GLU_SMOOTH)
    gluDisk(quadric, 0, 0.3, 16, 1)
    glEndList()

    glNewList(G_OBJ_GRID, GL_COMPILE)
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
    glBegin(GL_QUADS)
    for i in range(-200, 200):
        for j in range(-200, 200):
            x, z = i * 0.5, j * 0.5
            height_variation = (
                np.sin(x * 0.3) * np.cos(z * 0.3) * 0.2 +
                np.sin(x * 0.8) * np.cos(z * 0.8) * 0.1 +
                np.sin(x * 2.0) * np.cos(z * 2.0) * 0.05
            )

            h1 = height_variation
            h2 = (np.sin((x+0.5) * 0.3) * np.cos(z * 0.3) * 0.2 +
                  np.sin((x+0.5) * 0.8) * np.cos(z * 0.8) * 0.1 +
                  np.sin((x+0.5) * 2.0) * np.cos(z * 2.0) * 0.05)
            h3 = (np.sin((x+0.5) * 0.3) * np.cos((z+0.5) * 0.3) * 0.2 +
                  np.sin((x+0.5) * 0.8) * np.cos((z+0.5) * 0.8) * 0.1 +
                  np.sin((x+0.5) * 2.0) * np.cos((z+0.5) * 2.0) * 0.05)
            h4 = (np.sin(x * 0.3) * np.cos((z+0.5) * 0.3) * 0.2 +
                  np.sin(x * 0.8) * np.cos((z+0.5) * 0.8) * 0.1 +
                  np.sin(x * 2.0) * np.cos((z+0.5) * 2.0) * 0.05)

            base_whiteness = 0.95
            h1_color = base_whiteness + h1 * 0.05
            h2_color = base_whiteness + h2 * 0.05
            h3_color = base_whiteness + h3 * 0.05
            h4_color = base_whiteness + h4 * 0.05

            v1 = np.array([x, -1 + h1, z])
            v2 = np.array([x + 0.5, -1 + h2, z])
            v3 = np.array([x + 0.5, -1 + h3, z + 0.5])
            v4 = np.array([x, -1 + h4, z + 0.5])

            normal = np.cross(v2 - v1, v3 - v1)
            normal = normal / np.linalg.norm(normal)

            glNormal3f(normal[0], normal[1], normal[2])
            glColor3f(h1_color, h1_color, 1.0)
            glVertex3f(x, -1 + h1, z)
            glColor3f(h2_color, h2_color, 1.0)
            glVertex3f(x + 0.5, -1 + h2, z)
            glColor3f(h3_color, h3_color, 1.0)
            glVertex3f(x + 0.5, -1 + h3, z + 0.5)
            glColor3f(h4_color, h4_color, 1.0)
            glVertex3f(x, -1 + h4, z + 0.5)
    glEnd()
    glEndList()

    glNewList(G_OBJ_TREE, GL_COMPILE)
    quadric = gluNewQuadric()
    gluQuadricNormals(quadric, GLU_SMOOTH)

    glColor3f(0.25, 0.16, 0.1)
    glPushMatrix()
    glRotatef(-90, 1, 0, 0)
    gluCylinder(quadric, 0.4, 0.3, 4.5, 8, 1)
    glPopMatrix()

    heights = [2.5, 3.5, 4.5, 5.5, 6.5]
    sizes = [2.4, 2.0, 1.6, 1.2, 0.8]

    glColor3f(0.1, 0.25, 0.1)
    for h, s in zip(heights, sizes):
        glPushMatrix()
        glTranslatef(0, h, 0)
        glRotatef(-90, 1, 0, 0)
        gluCylinder(quadric, s, 0, 1.8, 8, 1)
        glPopMatrix()

    glColor3f(1.0, 1.0, 1.0)
    for h, s in zip(heights, sizes):

        glPushMatrix()
        glTranslatef(0, h + 0.2, 0)
        glRotatef(90, 1, 0, 0)
        gluCylinder(quadric, s * 0.9, s * 0.9, 0.04, 8, 1)
        glPopMatrix()
    glEndList()

    glNewList(G_OBJ_WALLS, GL_COMPILE)
    wall_size = 80
    wall_height = 50
    segments = 40

    glBegin(GL_TRIANGLES)
    for i in range(segments * 4):
        angle = i * (2 * np.pi / (segments * 4))
        next_angle = ((i + 1) % (segments * 4)) * (2 * np.pi / (segments * 4))

        x1 = wall_size * np.cos(angle)
        z1 = wall_size * np.sin(angle)
        x2 = wall_size * np.cos(next_angle)
        z2 = wall_size * np.sin(next_angle)

        h1 = wall_height + np.sin(angle * 4) * 2
        h2 = wall_height + np.sin(next_angle * 4) * 2

        glColor3f(0.85, 0.85, 0.95)

        glVertex3f(x1, -1, z1)
        glVertex3f(x2, -1, z2)
        glVertex3f(x1, h1, z1)

        glVertex3f(x1, h1, z1)
        glVertex3f(x2, h2, z2)
        glVertex3f(x2, -1, z2)

        glColor3f(1.0, 1.0, 1.0)
        glVertex3f(x1, h1 - 0.2, z1)
        glVertex3f(x2, h2 - 0.2, z2)
        glVertex3f(x1, h1, z1)
    glEnd()
    glEndList()

class SceneNode:
    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.rotation = [0, 0, 0]
        self.scale = [1, 1, 1]
        self.children = []
        self.movement_velocity = np.zeros(3)

    def add_child(self, child):
        self.children.append(child)

    def move_to(self, target, delta_time):

        for i in range(3):
            self.position[i], self.movement_velocity[i] = smooth_damp(
                self.position[i],
                target[i],
                self.movement_velocity[i],
                0.5,
                2.0,
                delta_time
            )

    def render(self):
        glPushMatrix()
        glTranslatef(*self.position)
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        glRotatef(self.rotation[2], 0, 0, 1)
        glScalef(*self.scale)
        self.render_self()
        for child in self.children:
            child.render()
        glPopMatrix()

    def render_self(self):
        pass

    def check_collision(self, pos, radius):

        dist = np.linalg.norm(pos - self.position)
        return dist < (radius + 1.0)

class Tree(SceneNode):
    def render_self(self):
        glCallList(G_OBJ_TREE)

class Snowman(SceneNode):
    def __init__(self):
        super().__init__()
        self.breath = 0
        self.scarf_wave = 0
        self.walk_cycle = 0
        self.facing_angle = 0
        self.target_facing_angle = 0
        self.angle_velocity = 0
        ground_height = self.get_ground_height(0, 0)
        self.position[1] = ground_height
        self.is_moving = False
        self.wander_timer = 0
        self.current_target = np.array([0.0, ground_height, 0.0])

    def render_self(self):
        glPushMatrix()
        glRotatef(self.facing_angle, 0, 1, 0)

        if self.is_moving:
            glTranslatef(0, np.sin(self.walk_cycle) * 0.03, 0)

        breath_scale = 1.0 + np.sin(self.breath) * 0.02

        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0.9, 0.9, 0.9, 1.0])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.05, 0.05, 0.05, 1.0])
        glMaterialf(GL_FRONT, GL_SHININESS, 1.0)

        glPushMatrix()
        glTranslatef(0, 0.25, 0)
        glScalef(0.25 * breath_scale, 0.25, 0.25 * breath_scale)
        gluSphere(gluNewQuadric(), 1, 8, 8)
        glPopMatrix()

        glPushMatrix()
        glTranslatef(0, 0.6, 0)
        glScalef(0.2 * breath_scale, 0.18, 0.2 * breath_scale)
        gluSphere(gluNewQuadric(), 1, 8, 8)
        glPopMatrix()

        glPushMatrix()
        glTranslatef(0, 0.88, 0)
        glScalef(0.15 * breath_scale, 0.15, 0.15 * breath_scale)
        gluSphere(gluNewQuadric(), 1, 8, 8)
        glPopMatrix()

        glColor3f(0.2, 0.2, 0.2)
        glPushMatrix()
        glTranslatef(0, 1.0, 0)
        glRotatef(-90, 1, 0, 0)
        gluDisk(gluNewQuadric(), 0, 0.2, 8, 1)
        gluCylinder(gluNewQuadric(), 0.1, 0.1, 0.15, 8, 1)
        glPopMatrix()

        glColor3f(0.1, 0.1, 0.1)
        for x in [-0.05, 0.05]:
            glPushMatrix()
            glTranslatef(x, 0.92, 0.12)
            glScalef(0.02, 0.02, 0.02)
            gluSphere(gluNewQuadric(), 1, 4, 4)
            glPopMatrix()

        glColor3f(1.0, 0.4, 0.0)
        glPushMatrix()
        glTranslatef(0, 0.9, 0.12)
        glRotatef(90, 1, 0, 0)
        gluCylinder(gluNewQuadric(), 0.025, 0.015, 0.08, 4, 1)
        glPopMatrix()

        glColor3f(0.8, 0.2, 0.2)
        glBegin(GL_QUADS)
        glVertex3f(-0.12, 0.8, 0.12)
        glVertex3f(0.12, 0.8, 0.12)
        glVertex3f(0.12, 0.7, 0.15)
        glVertex3f(-0.12, 0.7, 0.15)
        glEnd()

        glPopMatrix()

    def get_ground_height(self, x, z):
        height_variation = np.sin(x * 0.3) * np.cos(z * 0.3) * 0.1
        ground_height = -1.0 + height_variation
        return ground_height

    def update(self, delta_time, trees):
        super_pos = self.position.copy()

        ground_height = self.get_ground_height(self.position[0], self.position[2])
        self.position[1] = ground_height + 0.25

        self.breath = (self.breath + 1.0 * delta_time) % (2 * np.pi)
        self.scarf_wave = (self.scarf_wave + 1.5 * delta_time) % (2 * np.pi)

        self.wander_timer -= delta_time
        if self.wander_timer <= 0:
            angle = random.uniform(0, 2*np.pi)
            distance = random.uniform(5, 20)
            self.current_target = np.array([
                np.cos(angle) * distance,
                -0.6,
                np.sin(angle) * distance
            ])
            self.wander_timer = random.uniform(5, 10)

        dir_to_target = self.current_target - self.position
        distance_to_target = np.linalg.norm(dir_to_target)
        collision = False

        if distance_to_target > 0.1:
            self.is_moving = True

            self.target_facing_angle = math.degrees(math.atan2(dir_to_target[0], dir_to_target[2]))

            dir_to_target = dir_to_target / distance_to_target
            new_pos = self.position + dir_to_target * delta_time * 1.4

            for tree in trees:
                if np.linalg.norm(new_pos - tree.position) < 1.0:
                    collision = True
                    self.wander_timer = 0
                    break

            if not collision:
                self.position = new_pos
        else:
            self.is_moving = False

        ground_height = self.get_ground_height(self.position[0], self.position[2])
        self.position[1] = ground_height + 0.4

        if self.is_moving:
            self.walk_cycle = (self.walk_cycle + 4.0 * delta_time) % (2 * np.pi)
        else:
            self.walk_cycle = max(0, self.walk_cycle - 8.0 * delta_time)

        self.facing_angle, self.angle_velocity = smooth_damp(
            self.facing_angle,
            self.target_facing_angle,
            self.angle_velocity,
            0.1,
            360.0,
            delta_time
        )
        ground_height = self.get_ground_height(self.position[0], self.position[2])
        self.position[1] = ground_height

        if self.is_moving:
            self.walk_cycle = (self.walk_cycle + 4.0 * delta_time) % (2 * np.pi)
        else:
            self.walk_cycle = max(0, self.walk_cycle - 8.0 * delta_time)

import numpy as np
import random
from OpenGL.GL import *
from OpenGL.GLUT import *

class Bird(SceneNode):
    def __init__(self):
        super().__init__()
        self.max_radius = 70
        self.position = np.array([0.0, random.uniform(5, 50), 0.0])
        self.velocity = np.array([random.uniform(-2, 2), 0, random.uniform(-2, 2)])
        self.target_position = self.position.copy()
        self.resting_time = 0
        self.is_resting = False
        self.wing_angle = 0
        self.wing_flap_speed = 5.0
        self.wing_flap_amplitude = 35
        self.body_angle = random.uniform(0, 360)
        self.body_rotation_speed = 3.0
        self.max_speed = 6.0
        self.min_speed = 3
        self.current_height = self.position[1]
        self.target_height = random.uniform(4, 20)
        self.player_scare_distance = 6.0
        self.bob_amplitude = 0.15
        self.height_change_timer = random.uniform(5, 10)
        self.taking_off = False
        self.take_off_timer = 0
        self.gliding = False
        self.glide_blend = 0.0
        self.height_threshold = 1.0
        self.glide_descent_speed = 0.5
        self.glide_forward_boost = 1.2
        self.base_height = 0.3
        self.landing = False
        self.landing_start_pos = None
        self.landing_start_height = None
        self.landing_timer = 0
        self.landing_duration = 1.5
        self.animation_time = 0
        self.choose_new_target()

    def get_ground_height(self, x, z):
        height_variation = (
            np.sin(x * 0.3) * np.cos(z * 0.3) * 0.2 +
            np.sin(x * 0.8) * np.cos(z * 0.8) * 0.1 +
            np.sin(x * 2.0) * np.cos(z * 2.0) * 0.05
        )
        return -1.0 + height_variation

    def choose_new_target(self):
        if self.taking_off:
            forward = np.array([np.cos(np.radians(self.body_angle)), 0, np.sin(np.radians(self.body_angle))])
            self.target_position = self.position + forward * 15
            self.target_position[1] = random.uniform(5, 8)
        else:
            angle = random.uniform(0, 2 * np.pi)
            radius = random.uniform(0, self.max_radius * 0.7)
            target = np.array([
                radius * np.cos(angle),
                self.target_height,
                radius * np.sin(angle)
            ])

            dist_from_center = np.linalg.norm(target[[0, 2]])
            if dist_from_center > self.max_radius:
                target = target * (self.max_radius / dist_from_center)

            self.target_position = target

    def start_take_off(self, away_from_pos=None):
        if not self.is_resting:
            return

        self.is_resting = False
        self.taking_off = True
        self.take_off_timer = 0

        if away_from_pos is not None:
            direction = self.position - away_from_pos
            direction[1] = 0
            direction = direction / (np.linalg.norm(direction) + 0.0001)
            self.body_angle = np.degrees(np.arctan2(direction[2], direction[0]))

        forward = np.array([np.cos(np.radians(self.body_angle)), 0, np.sin(np.radians(self.body_angle))])
        self.velocity = forward * self.min_speed
        self.velocity[1] = 1.0
        self.choose_new_target()

    def start_landing(self):
        if self.is_resting or self.landing:
            return

        self.landing = True
        self.landing_timer = 0
        self.landing_start_pos = self.position.copy()
        self.landing_start_height = self.current_height
        ground_height = self.get_ground_height(self.position[0], self.position[2])

        forward = self.velocity.copy()
        forward[1] = 0
        forward = forward / (np.linalg.norm(forward) + 0.0001)
        landing_point = self.position + forward * 3
        landing_point[1] = ground_height + self.base_height
        self.target_position = landing_point

    def update(self, delta_time, trees, camera_pos):
        self.animation_time += delta_time

        if self.is_resting:
            ground_height = self.get_ground_height(self.position[0], self.position[2])
            self.position[1] = ground_height + self.base_height

            dist_to_player = np.linalg.norm(camera_pos - self.position)
            if dist_to_player < self.player_scare_distance:
                self.start_take_off(away_from_pos=camera_pos)
                return

            self.resting_time -= delta_time
            if self.resting_time <= 0:
                self.start_take_off()
            self.wing_angle = 0
            return

        if self.landing:
            self.landing_timer += delta_time
            progress = min(self.landing_timer / self.landing_duration, 1.0)

            if progress < 1.0:
                target_pos = self.target_position
                current_pos = self.landing_start_pos

                height_factor = np.sin((1 - progress) * np.pi/2)
                height_offset = 1.0 * height_factor

                self.position = current_pos * (1 - progress) + target_pos * progress
                self.position[1] += height_offset

                self.velocity *= (1 - progress * 0.1)

                if progress < 0.7:
                    base_flap = np.sin(self.animation_time * self.wing_flap_speed)
                    self.wing_angle = self.wing_flap_amplitude * base_flap * (1 - progress/0.7)
                else:
                    self.wing_angle = 0
            else:
                self.landing = False
                self.is_resting = True
                self.resting_time = random.uniform(5, 15)
                ground_height = self.get_ground_height(self.position[0], self.position[2])
                self.position[1] = ground_height + self.base_height
                self.wing_angle = 0
            return

        if self.taking_off:
            self.take_off_timer += delta_time

            if self.take_off_timer < 1.2:
                forward = np.array([np.cos(np.radians(self.body_angle)), 0, np.sin(np.radians(self.body_angle))])
                target_velocity = forward * self.min_speed
                target_velocity[1] = 1.5 * (1 - self.take_off_timer/1.2)

                self.velocity = self.velocity * 0.95 + target_velocity * 0.05
                self.position += self.velocity * delta_time

                base_flap = np.sin(self.animation_time * (self.wing_flap_speed * 1.5))
                self.wing_angle = self.wing_flap_amplitude * 1.2 * base_flap
            else:
                self.taking_off = False
                self.target_height = random.uniform(5, 8)
                self.current_height = self.position[1]
                self.choose_new_target()
            return

        height_diff = self.target_height - self.current_height

        if height_diff < -self.height_threshold:
            self.gliding = True
        elif abs(height_diff) <= self.height_threshold or height_diff > 0:
            self.gliding = False

        if self.gliding:
            self.glide_blend = min(1.0, self.glide_blend + delta_time * 2)
        else:
            self.glide_blend = max(0.0, self.glide_blend - delta_time * 2)

        dist_from_center = np.linalg.norm(self.position[[0, 2]])
        if dist_from_center > self.max_radius * 0.7:
            to_center = -self.position
            to_center[1] = 0
            to_center = to_center / (np.linalg.norm(to_center) + 0.0001)
            self.velocity = self.velocity * 0.9 + to_center * self.max_speed * 0.1
            self.choose_new_target()

        if random.random() < 0.001 and dist_from_center < self.max_radius * 0.5:
            self.start_landing()
            return

        to_target = self.target_position - self.position
        dist_to_target = np.linalg.norm(to_target)

        if dist_to_target < 3:
            if random.random() < 0.3 and dist_from_center < self.max_radius * 0.5:
                self.start_landing()
                return
            else:
                self.choose_new_target()
            return

        desired_direction = to_target / (dist_to_target + 0.0001)
        self.velocity = self.velocity * 0.98 + desired_direction * self.max_speed * 0.02

        if self.gliding:

            self.velocity[1] -= self.glide_descent_speed * delta_time * self.glide_blend
            forward = np.array([
                np.cos(np.radians(self.body_angle)),
                0,
                np.sin(np.radians(self.body_angle))
            ])
            self.velocity += forward * self.glide_forward_boost * delta_time * self.glide_blend
        else:
            self.velocity[1] += 0.1 * delta_time * (1 - self.glide_blend)

        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity *= self.max_speed / speed
        elif speed < self.min_speed and speed > 0:
            self.velocity *= self.min_speed / speed

        self.position += self.velocity * delta_time
        self.current_height += height_diff * delta_time

        if self.glide_blend > 0:
            glide_wing_angle = self.wing_flap_amplitude * 0.3
            flap_base = np.sin(self.animation_time * self.wing_flap_speed)
            flap_wing_angle = self.wing_flap_amplitude * flap_base

            self.wing_angle = glide_wing_angle * self.glide_blend + flap_wing_angle * (1 - self.glide_blend)
            bob_offset = np.sin(self.animation_time * self.wing_flap_speed) * self.bob_amplitude * (1 - self.glide_blend)
            self.position[1] = self.current_height + bob_offset
        else:
            base_flap = np.sin(self.animation_time * self.wing_flap_speed)
            self.wing_angle = self.wing_flap_amplitude * base_flap
            bob_offset = np.sin(self.animation_time * self.wing_flap_speed) * self.bob_amplitude
            self.position[1] = self.current_height + bob_offset

        if speed > 0.1:
            target_angle = np.degrees(np.arctan2(self.velocity[0], self.velocity[2]))
            angle_diff = (target_angle - self.body_angle + 180) % 360 - 180
            self.body_angle += angle_diff * self.body_rotation_speed * delta_time
            self.rotation[1] = self.body_angle

    def render_self(self):
        glColor3f(0.2, 0.2, 0.2)
        glPushMatrix()
        glScalef(1.0, 1.0, 1.0)
        glRotatef(90, 1, 0, 0)

        glPushMatrix()
        glScalef(0.6, 1.2, 0.6)
        glutSolidSphere(0.1, 10, 10)
        glPopMatrix()

        for side in [-1, 1]:
            glPushMatrix()
            glTranslatef(side * 0.07, 0, 0)
            glRotatef(side * (0 if self.is_resting else self.wing_angle), 0, 1, 0)
            glRotatef(10, 1, 0, 0)
            glScalef(0.6, 0.1, 1.4)
            glutSolidSphere(0.04, 8, 8)
            glPopMatrix()

        glPopMatrix()

class Rabbit(SceneNode):
    def __init__(self):
        super().__init__()
        self.max_radius = 70
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.target_position = self.position.copy()
        self.facing_angle = 0
        self.target_facing_angle = 0
        self.angle_velocity = 0
        self.is_hopping = False
        self.hop_height = 0
        self.hop_time = 0
        self.rest_time = random.uniform(2, 4)
        self.hop_duration = 0.5
        self.scale = [0.5, 0.5, 0.5]
        self.base_height = 0.3
        self.choose_new_target()

    def choose_new_target(self):
        angle = random.uniform(0, 2 * np.pi)
        distance = random.uniform(1, 2)
        self.target_position = self.position + np.array([
            np.cos(angle) * distance,
            0,
            np.sin(angle) * distance
        ])
        dist_from_center = np.linalg.norm(self.target_position[[0, 2]])
        if dist_from_center > self.max_radius:
            self.target_position = self.target_position * (self.max_radius / dist_from_center)

    def update(self, delta_time, trees, camera_pos):
        if not self.is_hopping:
            self.rest_time -= delta_time
            if self.rest_time <= 0:
                self.is_hopping = True
                self.hop_time = 0
                dir_to_target = self.target_position - self.position
                self.target_facing_angle = np.degrees(np.arctan2(dir_to_target[0], dir_to_target[2]))

        self.facing_angle, self.angle_velocity = smooth_damp(
            self.facing_angle,
            self.target_facing_angle,
            self.angle_velocity,
            0.1,
            360.0,
            delta_time
        )

        if self.is_hopping:
            self.hop_time += delta_time
            progress = self.hop_time / self.hop_duration

            if progress <= 1:
                hop_progress = np.sin(progress * np.pi)
                hop_adjust = -(progress * progress) + progress
                final_hop = (hop_progress + hop_adjust) * 0.5

                start_pos = self.position.copy()
                dir_to_target = self.target_position - start_pos
                self.position = start_pos + dir_to_target * progress
                self.hop_height = final_hop * 0.3
                ground_height = self.get_ground_height(self.position[0], self.position[2])
                self.position[1] = ground_height + self.base_height + self.hop_height
            else:
                self.is_hopping = False
                self.hop_height = 0
                self.rest_time = random.uniform(2, 4)
                self.choose_new_target()

        ground_height = self.get_ground_height(self.position[0], self.position[2])
        self.position[1] = ground_height + self.base_height + self.hop_height

    def get_ground_height(self, x, z):
        height_variation = (
            np.sin(x * 0.3) * np.cos(z * 0.3) * 0.2 +
            np.sin(x * 0.8) * np.cos(z * 0.8) * 0.1 +
            np.sin(x * 2.0) * np.cos(z * 2.0) * 0.05
        )
        return -1.0 + height_variation

    def render_self(self):
        glPushMatrix()
        glRotatef(self.facing_angle, 0, 1, 0)

        body_color = [0.6, 0.6, 0.6]

        glColor3f(*body_color)
        glPushMatrix()
        glScalef(1.0, 0.8, 1.2)
        glutSolidSphere(0.3, 12, 12)
        glPopMatrix()

        glPushMatrix()
        glTranslatef(0, 0.2, 0.3)
        glutSolidSphere(0.15, 10, 10)

        for x in [-0.08, 0.08]:
            glPushMatrix()
            glTranslatef(x, 0.15, 0)
            glScalef(0.05, 0.3, 0.05)
            glutSolidCube(1.0)
            glPopMatrix()
        glPopMatrix()

        glPushMatrix()
        glTranslatef(0, 0.1, -0.4)
        glutSolidSphere(0.08, 8, 8)
        glPopMatrix()

        foot_hop = 0.1 * self.hop_height if self.is_hopping else 0
        for x in [-0.15, 0.15]:
            for z in [-0.2, 0.2]:
                glPushMatrix()
                glTranslatef(x, -0.2 - foot_hop, z)
                glScalef(0.1, 0.1, 0.15)
                glutSolidCube(1.0)
                glPopMatrix()

        glPopMatrix()

class Viewer:
    def __init__(self):
        glutInit()
        glutInitWindowSize(1024, 768)
        glutCreateWindow("Magical Winter Scene")
        (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutDisplayFunc(self.render)
        glutKeyboardFunc(self.keyboard)
        glutKeyboardUpFunc(self.keyboard_up)
        glutReshapeFunc(self.reshape)

        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_pressed = False
        self.camera_yaw = 0
        self.camera_pitch = 0
        glutMotionFunc(self.mouse_motion)
        glutMouseFunc(self.mouse_button)

        self.camera_x = 0.0
        self.camera_height = 2.0
        self.camera_z = 0.0
        self.movement = {'w': False, 's': False, 'a': False, 'd': False, 'q': False, 'e': False}
        self.camera_velocity = np.zeros(3)

        self.gravity_enabled = True
        self.vertical_velocity = 0
        self.is_grounded = True

        self.last_time = time()
        self.day_cycle = 0.0
        self.frame_count = 0
        self.fps = 0
        self.fps_time = time()

        self.scene_root = SceneNode()
        self.snowman = Snowman()
        self.scene_root.add_child(self.snowman)

        self.birds = []
        for _ in range(50):
            bird = Bird()
            bird.position = np.array([
                random.uniform(-95, 95),
                random.uniform(5, 20),
                random.uniform(-95, 95)
            ])
            bird.target_position = np.array([
                random.uniform(-95, 95),
                random.uniform(5, 20),
                random.uniform(-95, 95)
            ])
            self.scene_root.add_child(bird)
            self.birds.append(bird)

        self.rabbits = []
        for _ in range(15):
            rabbit = Rabbit()
            rabbit.position = np.array([
                random.uniform(-60, 60),
                0,
                random.uniform(-60, 60)
            ])
            rabbit.position[1] = self.get_height(rabbit.position[0], rabbit.position[2])
            self.scene_root.add_child(rabbit)
            self.rabbits.append(rabbit)

        self.trees = []
        min_tree_distance = 3.0
        wall_buffer = 25.0
        max_tree_radius = 95 - wall_buffer

        def is_position_valid(pos_x, pos_z, existing_trees):

            for existing_tree in existing_trees:
                dist = np.sqrt((pos_x - existing_tree.position[0])**2 +
                              (pos_z - existing_tree.position[2])**2)
                if dist < min_tree_distance:
                    return False

            dist_from_center = np.sqrt(pos_x**2 + pos_z**2)
            if dist_from_center > max_tree_radius:
                return False

            return True

        for _ in range(20):
            tree = Tree()

            angle = random.uniform(0, 2*np.pi)
            dist = random.uniform(5, max_tree_radius)
            pos_x = np.cos(angle)*dist
            pos_z = np.sin(angle)*dist

            if not is_position_valid(pos_x, pos_z, self.trees):
                continue

            height_variation = (
                np.sin(pos_x * 0.3) * np.cos(pos_z * 0.3) * 0.2 +
                np.sin(pos_x * 0.8) * np.cos(pos_z * 0.8) * 0.1 +
                np.sin(pos_x * 2.0) * np.cos(pos_z * 2.0) * 0.05
            )

            tree.position = np.array([
                pos_x,
                -1.0 + height_variation,
                pos_z
            ])
            tree.rotation[1] = random.uniform(0, 360)
            scale = random.uniform(0.8, 1.4)
            tree.scale = [scale, scale + random.uniform(-0.1, 0.1), scale]
            self.trees.append(tree)
            self.scene_root.add_child(tree)

        for _ in range(300):
            tree = Tree()

            angle = random.uniform(0, 2*np.pi)
            dist = random.uniform(5, 95)
            pos_x = np.cos(angle)*dist
            pos_z = np.sin(angle)*dist

            if abs(pos_x) > 95 or abs(pos_z) > 95:
                continue

            if not is_position_valid(pos_x, pos_z, self.trees):
                continue

            height_variation = (
                np.sin(pos_x * 0.3) * np.cos(pos_z * 0.3) * 0.2 +
                np.sin(pos_x * 0.8) * np.cos(pos_z * 0.8) * 0.1 +
                np.sin(pos_x * 2.0) * np.cos(pos_z * 2.0) * 0.05
            )

            tree.position = np.array([
                pos_x,
                -1.0 + height_variation,
                pos_z
            ])
            tree.rotation[1] = random.uniform(0, 360)
            scale = random.uniform(0.8, 1.4)
            tree.scale = [scale, scale + random.uniform(-0.1, 0.1), scale]
            self.trees.append(tree)
            self.scene_root.add_child(tree)

            if random.random() < 0.2:
                for _ in range(random.randint(2, 3)):
                    cluster_tree = Tree()

                    for _ in range(10):
                        offset_dist = random.uniform(min_tree_distance, min_tree_distance + 2)
                        offset_angle = random.uniform(0, 2*np.pi)
                        cluster_x = pos_x + np.cos(offset_angle) * offset_dist
                        cluster_z = pos_z + np.sin(offset_angle) * offset_dist

                        if abs(cluster_x) > 95 or abs(cluster_z) > 95:
                            continue

                        if is_position_valid(cluster_x, cluster_z, self.trees):
                            cluster_height = (
                                np.sin(cluster_x * 0.3) * np.cos(cluster_z * 0.3) * 0.2 +
                                np.sin(cluster_x * 0.8) * np.cos(cluster_z * 0.8) * 0.1 +
                                np.sin(cluster_x * 2.0) * np.cos(cluster_z * 2.0) * 0.05
                            )

                            cluster_tree.position = np.array([
                                cluster_x,
                                -1.0 + cluster_height,
                                cluster_z
                            ])
                            cluster_tree.rotation[1] = random.uniform(0, 360)
                            scale = random.uniform(0.7, 1.1)
                            cluster_tree.scale = [scale, scale + random.uniform(-0.1, 0.1), scale]
                            self.trees.append(cluster_tree)
                            self.scene_root.add_child(cluster_tree)
                            break

        self.init_snow_vbo()

    def init_snow_vbo(self):
        self.num_particles = 4000
        world_size = 160

        particle_data = np.zeros(self.num_particles,
            dtype=[('position', np.float32, 3),
                    ('phase', np.float32),
                    ('speed', np.float32),
                    ('size', np.float32)])

        particle_data['position'][:, 0] = np.random.uniform(-world_size/2, world_size/2, self.num_particles)
        particle_data['position'][:, 1] = np.random.uniform(0, 50, self.num_particles)
        particle_data['position'][:, 2] = np.random.uniform(-world_size/2, world_size/2, self.num_particles)
        particle_data['phase'] = np.random.uniform(0, 2 * np.pi, self.num_particles)
        particle_data['speed'] = np.random.uniform(1.5, 4.5, self.num_particles)
        particle_data['size'] = np.random.uniform(3.0, 6.0, self.num_particles)

        self.snow_vbo = vbo.VBO(particle_data, usage='GL_DYNAMIC_DRAW')

        self.particle_positions = particle_data['position']
        self.particle_phases = particle_data['phase']
        self.particle_speeds = particle_data['speed']
        self.particle_sizes = particle_data['size']

        glutIdleFunc(self.idle)
        self.init_opengl()
        init_primitives()

    def check_wall_collision(self, new_x, new_z):
        WALL_SIZE = 78
        BUFFER = 2.0

        r = np.sqrt(new_x * new_x + new_z * new_z)

        if r > WALL_SIZE:

            angle = np.arctan2(new_z, new_x)

            slide_x = WALL_SIZE * np.cos(angle)
            slide_z = WALL_SIZE * np.sin(angle)
            return True, (slide_x, slide_z)

        return False, (new_x, new_z)

    def mouse_motion(self, x, y):
        if self.mouse_pressed:
            dx = x - self.mouse_x
            dy = y - self.mouse_y

            self.camera_yaw += dx * 0.1

            self.camera_pitch = np.clip(self.camera_pitch + dy * 0.1, -90, 90)

            self.mouse_x = x
            self.mouse_y = y
            glutPostRedisplay()

    def mouse_button(self, button, state, x, y):
        if button == GLUT_LEFT_BUTTON:
            if state == GLUT_DOWN:
                self.mouse_pressed = True
                self.mouse_x = x
                self.mouse_y = y
            else:
                self.mouse_pressed = False

    def init_opengl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glMatrixMode(GL_PROJECTION)
        glMatrixMode(GL_MODELVIEW)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        gluPerspective(45.0, 1024.0/768.0, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.1, 0.1, 0.1, 1.0])
        glMaterialf(GL_FRONT, GL_SHININESS, 8.0)

        glEnable(GL_FOG)
        glFogfv(GL_FOG_COLOR, [0.7, 0.8, 0.9, 1.0])
        glFogi(GL_FOG_MODE, GL_EXP2)
        glFogf(GL_FOG_DENSITY, 0.025)
        glFogf(GL_FOG_START, 10.0)
        glFogf(GL_FOG_END, 60.0)

        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.4, 0.4, 0.5, 1.0])

    def keyboard(self, key, x, y):
        key = key.decode('utf-8').lower()
        if key in self.movement:
            self.movement[key] = True
        elif key == 'f':
            self.gravity_enabled = not self.gravity_enabled
            if not self.gravity_enabled:
                self.vertical_velocity = 0

    def keyboard_up(self, key, x, y):
        key = key.decode('utf-8').lower()
        if key in self.movement:
            self.movement[key] = False

    def update_camera(self, delta_time):
        forward = np.array([
            np.sin(np.radians(self.camera_yaw)),
            0,
            np.cos(np.radians(self.camera_yaw))
        ])
        right = np.array([
            np.cos(np.radians(self.camera_yaw)),
            0,
            -np.sin(np.radians(self.camera_yaw))
        ])

        move_forward = float(self.movement['s']) - float(self.movement['w'])
        move_right = float(self.movement['d']) - float(self.movement['a'])
        move_up = float(self.movement['q']) - float(self.movement['e'])

        move_speed = 50.0
        move_vector = forward * move_forward + right * move_right

        ground_height = (
            np.sin(self.camera_x * 0.3) * np.cos(self.camera_z * 0.3) * 0.2 +
            np.sin(self.camera_x * 0.8) * np.cos(self.camera_z * 0.8) * 0.1 +
            np.sin(self.camera_x * 2.0) * np.cos(self.camera_z * 2.0) * 0.05
        )
        absolute_ground = -1.0 + ground_height

        if self.gravity_enabled:
            GRAVITY = 9.8
            self.vertical_velocity -= GRAVITY * delta_time
            new_height = self.camera_height + self.vertical_velocity * delta_time

            if new_height <= absolute_ground + 1.1:
                new_height = absolute_ground + 1.1
                self.vertical_velocity = 0
                self.is_grounded = True
            else:
                self.is_grounded = False

            self.camera_height = new_height
        else:
            move_vector[1] = move_up
            new_pos = np.array([self.camera_x, self.camera_height, self.camera_z])
            for i in range(3):
                new_pos[i], self.camera_velocity[i] = smooth_damp(
                    new_pos[i],
                    new_pos[i] + move_vector[i] * move_speed * delta_time,
                    self.camera_velocity[i],
                    0.1,
                    10.0,
                    delta_time
                )
            self.camera_height = new_pos[1]

        new_pos = np.array([self.camera_x, 0, self.camera_z])
        move_vector[1] = 0

        potential_x = self.camera_x
        potential_z = self.camera_z

        for i in [0, 2]:
            new_value, self.camera_velocity[i] = smooth_damp(
                new_pos[i],
                new_pos[i] + move_vector[i] * move_speed * delta_time,
                self.camera_velocity[i],
                0.1,
                10.0,
                delta_time
            )
            if i == 0:
                potential_x = new_value
            else:
                potential_z = new_value

        collides, (new_x, new_z) = self.check_wall_collision(potential_x, potential_z)
        self.camera_x = new_x
        self.camera_z = new_z

        min_height = absolute_ground + 1.2
        self.camera_height = max(min_height, self.camera_height)

    def update_snow(self, delta_time):
        world_size = 160
        half_size = world_size / 2
        time_offset = time() * 0.5

        wind = np.sin(time_offset + self.particle_phases) * 0.02
        self.particle_positions[:, 1] -= self.particle_speeds * 0.6 * delta_time
        self.particle_positions[:, 0] += wind

        reset_mask = ((self.particle_positions[:, 1] < -1.0) |
                        (np.abs(self.particle_positions[:, 0]) > half_size) |
                        (np.abs(self.particle_positions[:, 2]) > half_size))

        num_reset = np.sum(reset_mask)
        if num_reset > 0:
            self.particle_positions[reset_mask, 1] = 50
            self.particle_positions[reset_mask, 0] = np.random.uniform(-half_size, half_size, num_reset)
            self.particle_positions[reset_mask, 2] = np.random.uniform(-half_size, half_size, num_reset)

        particle_data = np.zeros(self.num_particles,
            dtype=[('position', np.float32, 3),
                    ('phase', np.float32),
                    ('speed', np.float32),
                    ('size', np.float32)])

        particle_data['position'] = self.particle_positions
        particle_data['phase'] = self.particle_phases
        particle_data['speed'] = self.particle_speeds
        particle_data['size'] = self.particle_sizes

        self.snow_vbo.set_array(particle_data)

    def get_height(self, x, z):

        if -200 <= x < 200 and -200 <= z < 200:
            height_variation = (
                np.sin(x * 0.3) * np.cos(z * 0.3) * 0.2 +
                np.sin(x * 0.8) * np.cos(z * 0.8) * 0.1 +
                np.sin(x * 2.0) * np.cos(z * 2.0) * 0.05
            )
            return -1 + height_variation
        else:

            return -1

    def render_snow(self):
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glEnable(GL_POINT_SPRITE)
        glEnable(GL_POINT_SMOOTH)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        dx = self.particle_positions[:, 0] - self.camera_x
        dz = self.particle_positions[:, 2] - self.camera_z
        distances = np.sqrt(dx * dx + dz * dz)
        view_mask = distances <= 60

        if np.any(view_mask):
            self.snow_vbo.bind()

            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 24, self.snow_vbo)

            visible_indices = np.where(view_mask)[0]
            for idx in visible_indices:
                distance = distances[idx]
                alpha = 0.8 * (1.0 - distance / 60)
                size = self.particle_sizes[idx]

                glPointSize(size)
                glColor4f(1, 1, 1, alpha)
                glDrawArrays(GL_POINTS, idx, 1)

            glDisableClientState(GL_VERTEX_ARRAY)
            self.snow_vbo.unbind()

        glDisable(GL_POINT_SMOOTH)
        glDisable(GL_POINT_SPRITE)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

    def render_scanlines(self):
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, 1024, 0, 768, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glColor4f(0, 0, 0, 0.1)
        glBegin(GL_LINES)
        for y in range(0, 768, 2):
            glVertex2f(0, y)
            glVertex2f(1024, y)
        glEnd()

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()

        glMatrixMode(GL_MODELVIEW)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glDisable(GL_BLEND)

    def idle(self):
        current_time = time()
        delta_time = current_time - self.last_time
        self.last_time = current_time

        self.frame_count += 1
        if current_time - self.fps_time > 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.fps_time = current_time

        self.day_cycle = (self.day_cycle + 0.1 * delta_time) % (2*np.pi)

        self.update_camera(delta_time)

        self.snowman.update(delta_time, self.trees)

        for bird in self.birds:
            bird.update(delta_time, self.trees, np.array([self.camera_x, self.camera_height, self.camera_z]))

        for rabbit in self.rabbits:
            rabbit.update(delta_time, self.trees, np.array([self.camera_x, self.camera_height, self.camera_z]))

        self.update_snow(delta_time)

        glutPostRedisplay()

    def render(self):

        day_color = [0.7, 0.8, 0.9, 1.0]
        sunset_color = [0.8, 0.5, 0.7, 1.0]
        night_color = [0.1, 0.1, 0.2, 1.0]

        t = (np.sin(self.day_cycle) + 1) / 2

        if t < 0.5:

            sky_color = [
                lerp(night_color[0], sunset_color[0], t * 2),
                lerp(night_color[1], sunset_color[1], t * 2),
                lerp(night_color[2], sunset_color[2], t * 2),
                1.0
            ]
        else:

            sky_color = [
                lerp(sunset_color[0], day_color[0], (t - 0.5) * 2),
                lerp(sunset_color[1], day_color[1], (t - 0.5) * 2),
                lerp(sunset_color[2], day_color[2], (t - 0.5) * 2),
                1.0
            ]

        glClearColor(*sky_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glRotatef(-self.camera_pitch, 1, 0, 0)
        glRotatef(-self.camera_yaw, 0, 1, 0)
        glTranslatef(-self.camera_x, -self.camera_height, -self.camera_z)

        ambient_light = [sky_color[0] * 0.2, sky_color[1] * 0.2, sky_color[2] * 0.2, 1.0]
        diffuse_light = [sky_color[0] * 0.8, sky_color[1] * 0.8, sky_color[2] * 0.8, 1.0]

        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient_light)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse_light)
        glLightfv(GL_LIGHT0, GL_SPECULAR, [diffuse_light[0] * 0.5, diffuse_light[1] * 0.5, diffuse_light[2] * 0.5, 1.0])

        sun_height = np.sin(self.day_cycle)
        glLightfv(GL_LIGHT0, GL_POSITION, [1, sun_height + 1, 1, 0])
        glLightfv(GL_LIGHT1, GL_POSITION, [-1, 0.5, -1, 0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.15, 0.15, 0.2, 1.0])
        glLightfv(GL_LIGHT1, GL_SPECULAR, [0, 0, 0, 1.0])

        fog_color = [sky_color[0] * 0.8, sky_color[1] * 0.8, sky_color[2] * 0.8, 1.0]
        glFogfv(GL_FOG_COLOR, fog_color)

        glCallList(G_OBJ_GRID)
        glCallList(G_OBJ_WALLS)

        self.render_scene(self.scene_root)
        self.render_snow()
        self.render_scanlines()

        self.render_hud()

        glutSwapBuffers()

    def render_scene(self, node):
        node.render()
        for child in node.children:
            self.render_scene(child)

    def render_hud(self):

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, 1024, 768, 0, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glColor4f(0, 0, 0, 0.5)
        glBegin(GL_QUADS)
        glVertex2f(10, 10)
        glVertex2f(250, 10)
        glVertex2f(250, 150)
        glVertex2f(10, 150)
        glEnd()

        glColor3f(1, 1, 1)
        draw_text(20, 25, "WASD - Move around")
        draw_text(20, 45, "Q/E - Adjust Camera Height")
        draw_text(20, 65, "F - Toggle Gravity")
        draw_text(20, 85, "Mouse - Adjust view")
        time_of_day = "Day" if np.sin(self.day_cycle) > 0 else "Night"
        draw_text(20, 105, f"Time: {time_of_day}")
        draw_text(20, 125, f"Gravity: {'On' if self.gravity_enabled else 'Off'}")
        draw_text(20, 145, f"FPS: {self.fps}")

        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, float(1024)/768, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def reshape(self, width, height):
        if height == 0:
            height = 1

        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        aspect_ratio = float(width) / height

        gluPerspective(45.0, aspect_ratio, 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

def main():
    viewer = Viewer()
    glutMainLoop()

if __name__ == "__main__":
    main()
