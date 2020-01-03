# Combined_synthesis_planning

This is the code repository for the paper 
Combining Retrosynthesis and Mixed-Integer Optimization for Minimizing the Chemical Inventory Needed to Realize a WHO Essential Medicines List

The code contains two parts - the script for retrosynthesis for multiple targets, and the script for optimization and results visualization.

## Retrosynthesis
tree_builder_with_evaluation_for_multiple_targets.py runs retrosynthesis planning for multiple targets. It calls the tree builder, condition recommender and reaction evaluator from the open source version of ASKCOS, which is available at https://github.com/connorcoley/ASKCOS. This script is made available now for reference, and I am working on facilitating the integration of this script with ASKCOS for easier user access.

## Optimization
The trees found for the 99 targets in the WHO essential medicines list are provided as a separate data file, so the users can test the optimization scripts independent of the retrosynthetic tools. The files are at .

There are two jupyter notebook files. optimization_of_pathway_selection.ipynb is for loading data and running the optimization. External dependencies include pandas, rdkit, pulp and cplex (which can be substituted with other solvers). 
Analyze_results.ipynb is for results summary and creating the html file for visualization. 

Currently these two scripts are to some extent mixed, i.e., the optimization script has some results analysis in it as well. 

tree_visualization_template.html is a template html file. The user need to mannually copy paste the tree data into the file for the visualization. Analyze_results.ipynb will be updated to make this process automatic.

