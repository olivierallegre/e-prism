from .knowledge_components import KnowledgeComponent


class PrerequisiteLink:

    def __init__(self, source, target, strength='strong'):
        """
        Prerequisite link object between two KCs.
        :param source: KnowledgeComponent, source of the prerequisite link
        :param target: KnowledgeComponent, target of the prerequisite link
        :param strength: str, strength of the prerequisite link
        """
        assert all((isinstance(source, KnowledgeComponent), isinstance(target, KnowledgeComponent))), \
            "Source and target KCs must be mskt KnowledgeComponent objects."
        self.source = source
        self.target = target
        assert strength in ('none', 'strong', 'weak')
        self.strength = strength

    def set_strength(self, strength):
        """
        Set the strength of the prerequisite link to given strength
        :param strength: new strength
        :return:
        """
        self.strength = strength

    def get_strength(self):
        """
        Get the strength of the prerequisite link.
        :return: Strength of the link
        """
        return self.strength


class SetOfPrerequisiteLinks:

    def __init__(self, prerequisite_links):
        """
        Set of prerequisite links.
        """
        self.prerequisite_links = prerequisite_links
        self.causal_map = {}
        for link in self.prerequisite_links:
            if link.source not in self.causal_map.keys():
                self.causal_map[link.source] = {}
            self.causal_map[link.source][link.target] = link
        self.anticausal_map = {}
        for link in self.prerequisite_links:
            if link.target not in self.anticausal_map.keys():
                self.anticausal_map[link.target] = {}
            self.anticausal_map[link.target][link.source] = link

    def __str__(self):
        return [f'{link.source.name}->{link.target.name}' for link in self.prerequisite_links]

    def get_prerequisite_link(self, source_kc, target_kc):
        """
        Return a prerequisite link of the set of prerequisite link from its source and target KCs.
        :param source_kc: the source KC of the prerequisite link
        :param target_kc: the target KC of the prerequisite link
        :return: the prerequisite link of the set of prerequisite link that has both source_kc as source and target_kc
        as target, else None
        """
        if target_kc in self.anticausal_map.keys():
            if source_kc in self.anticausal_map[target_kc].keys():
                return self.anticausal_map[target_kc][source_kc]
        return None

    def add_prerequisite_link(self, source_kc, target_kc, strength):
        if self.get_prerequisite_link(source_kc, target_kc) is None:
            self.anticausal_map[target_kc][source_kc] = PrerequisiteLink(source_kc, target_kc, strength)

    def set_prerequisite_link_strength(self, source_kc, target_kc, strength):
        """
        Set the strength of a prerequisite link of the set of prerequisite link to a given value.
        :param source_kc: the source KC of the link that should be changed
        :param target_kc: the target KC of the link that should be changed
        :param strength: str, the new value of the strength
        """
        assert strength in ['strong', 'weak']
        link_existence = False
        if target_kc in self.anticausal_map.keys():
            if source_kc in self.anticausal_map[target_kc].keys():
                self.anticausal_map[target_kc][source_kc].set_strength(strength)
                link_existence = True

        if not link_existence:
            self.add_prerequisite_link(source_kc, target_kc, strength)

    def get_prerequisite_link_strength(self, source_kc, target_kc):
        """
        Return the strength of a prerequisite link of the set of prerequisite links.
        :param source_kc: the source KC of the prerequisite link
        :param target_kc: the target KC of the prerequisite link
        :return: str, the strength of the prerequisite link
        """
        if target_kc in self.anticausal_map.keys():
            if source_kc in self.anticausal_map[target_kc].keys():
                return self.anticausal_map[target_kc][source_kc].get_strength()
        return 'not existing'

    def get_kc_parents(self, kc):
        """
        Return the parents of a given KC in the set of prerequisite links.
        :param kc: KnowledgeComponent
        :return: list, parents of the given KC
        """
        if kc in self.anticausal_map.keys():
            return [parent for parent in self.anticausal_map[kc].keys() if (
                    self.get_prerequisite_link_strength(parent, kc) != 'not existing')]
        else:
            return []

    def get_kc_children(self, kc):
        """
        Return the children of a given KC in the set of prerequisite links.
        :param kc: KnowledgeComponent
        :return: list, children of the given KC
        """
        return [child for child in self.anticausal_map.keys() if (
                (kc in list(self.anticausal_map[child].keys())) and
                (self.get_prerequisite_link_strength(kc, child) != 'not existing'))
                ]

    def get_anticausal_map(self):
        return self.anticausal_map

    def get_causal_map(self):
        return self.causal_map

    def get_recursive_parents(self, root_kc):
        """
        Return the list of every recursive parent of a given KC (that is to say parents of KC, parents of every parent
        of KC, and so on).
        :param root_kc: the root KC
        :return: list of every recursive parent of KC.
        """
        parents = []

        def _get_parents_leaf_nodes(kc):
            if kc is not None:
                if len(self.get_kc_parents(kc)) == 0:
                    parents.append(kc)
                for parent in self.get_kc_parents(kc):
                    _get_parents_leaf_nodes(parent)

        _get_parents_leaf_nodes(root_kc)
        return parents

    def get_recursive_children(self, root_kc):
        """
        Return the list of every recursive child of a given KC (that is to say children of KC, children of every child
        of KC, and so on).
        :param root_kc: the root KC
        :return: list of every recursive child of KC.
        """
        children = []

        def _get_children_leaf_nodes(kc):
            if kc is not None:
                if len(self.get_kc_children(kc)) == 0:
                    children.append(kc)
                for child in self.get_kc_children(kc):
                    _get_children_leaf_nodes(child)

        _get_children_leaf_nodes(root_kc)
        return children

    def to_csv(self, csv_path):
        import pandas as pd
        data = [[link.source.id, link.target.id] for link in self.prerequisite_links]
        df = pd.DataFrame(data, columns=["source", "target"])
        return df.to_csv(csv_path)
