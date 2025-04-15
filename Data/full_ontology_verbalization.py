def full_ontology_injection(ontology):
    branch_info = []

    # Helper function to format the class and its relations
    def format_class_relations(cls_list, indent=0):
        relations = []

        # Classes
        for cls in cls_list:

            # # Extract superclasses
            # superclasses = cls.ancestors()
            # if superclasses != cls:
            #     relations.append("  " * (indent + 1) + f"Superclasses: {', '.join([s.name for s in superclasses if hasattr(s, 'name')])}")

            # Class name
            relations.append("  " * (indent + 2) + f"Class: {cls.name}")

            # Help function to iteratively collect all subclasses
            subclasses = []
            def collect_all_subclasses(cls, subclasses_list, current_indent=0):
                direct_subclasses = list(cls.subclasses())
                for subclass in direct_subclasses:
                    subclasses_list.append((subclass, current_indent))
                    collect_all_subclasses(subclass, subclasses_list, current_indent + 1)

            # Subclasses
            collect_all_subclasses(cls, subclasses, indent + 3)
            for subclass, sub_indent in subclasses:
                relations.append("  " * sub_indent + f"Subclass: {subclass.name}")

                # Lexical forms
                lex_annotations = getattr(subclass, "lex", [])
                if lex_annotations:
                    relations.append("  " * (sub_indent + 1)+ f"Lexical Forms: {', '.join(lex_annotations)}")

                # Sentiment relations
                if "negative" in subclass.name.lower():
                    relations.append("  " * (sub_indent + 1) + f"Sentiment Relation: Negative")
                elif "positive" in subclass.name.lower():
                    relations.append("  " * (sub_indent + 1) + f"Sentiment Relation: Positive")
                elif "neutral" in subclass.name.lower():
                    relations.append("  " * (sub_indent + 1) + f"Sentiment Relation: Neutral")

            # Space
            relations.append("")

        return relations

    # Function to check if a class is a root class
    def is_root_class(cls):
        # A class is a root if it has no parents (is_a is empty) or only owl.Thing as parent
        parents = cls.is_a
        return not parents or (len(parents) == 1 and parents[0] == owl.Thing)

    root_classes = []
    # Find root classes
    for cls in ontology.classes():
        if is_root_class(cls) and 'sentiment' not in cls.name.lower():
            root_classes.append(cls)
    branch_info.extend(format_class_relations(root_classes))

    # Return extraced branches as a formatted string
    return "\n".join(branch_info)

# To run the code:
# - pip install owlready2
# - from owlready2 import *

# Main
domain_ontology_path = 
ontology = get_ontology(f"file://{domain_ontology_path}").load()
ontology_injection = full_ontology_injection(ontology)
print(ontology_injection)
