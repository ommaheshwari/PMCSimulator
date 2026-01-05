from Modules.Settings.settings import *

def add_reaction(particles, materials, type, Reactions,count):
    for particle in particles:
        for layer_material in materials:
            if 's' in type:
                R = "{}<g> + {}<s> = {}<q> ".format(particle, layer_material, layer_material)
                name = "sputter_"+str(layer_material)+"_"+str(particle)+"_neutral"
                Reactions['Sputter'][name]=R
                R = "{}+<g> + {}<s> = {}<q> ".format(particle, layer_material, layer_material)
                name = "sputter_"+str(layer_material)+"_"+str(particle)+"_ion"
                Reactions['Sputter'][name]=R
                count=count+2
            if 'r' in type:
                R = "{}<g> + {}<s> = {}<s> + {}<r> ".format(particle, layer_material, layer_material, particle)
                name = "reflect_"+str(layer_material)+"_"+str(particle)+"_neutral"
                Reactions['Reflection'][name]=R                
                R = "{}+<g> + {}<s> = {}<s> + {}+<r> ".format(particle, layer_material, layer_material, particle)
                name = "reflect_"+str(layer_material)+"_"+str(particle)+"_ion"
                Reactions['Reflection'][name]=R
                count=count+2
            if 'c' in type:
                R = "{}<g> + {}<s> = {}<q> ".format(particle, layer_material, layer_material)
                name = "chemical_etch_"+str(layer_material)+"_"+str(particle)+"_neutral"
                Reactions['Chemical_Etch'][name]=R                
                R = "{}+<g> + {}<s> = {}<q> ".format(particle, layer_material, layer_material)
                name = "chemical_etch_"+str(layer_material)+"_"+str(particle)+"_ion"
                Reactions['Chemical_Etch'][name]=R
                count=count+2
    return Reactions, count
