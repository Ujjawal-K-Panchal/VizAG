"""
Title: Project Configuration.

Date: May 27, 2024; 7:22 PM

Author: Ujjawal K. Panchal & Ajinkya Chaudhari & Isha S. Joglekar
"""
import os

modelstore = "./modelstore"

if not os.path.exists(modelstore):
    os.makedirs(modelstore)