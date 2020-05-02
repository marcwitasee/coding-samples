'''
CAPP 30122 W'20: Markov models and hash tables

Marc Richardson
'''

TOO_FULL = 0.5
GROWTH_RATIO = 2


class HashEntry:
    '''Class for representing a cell in a hash table'''

    def __init__(self, key, val):
        '''
        Construct a HashTable entry with a key (str) and value
        '''

        self.key = key
        self.val = val


    def __repr__(self):
        '''
        Representation of HashEntry object
        '''

        return "({}, {})".format(self.key, self.val)


class HashTable:
    '''Class for representing a hash table'''

    def __init__(self, cells, defval):
        '''
        Construct a new hash table with a fixed number of cells equal to the
        parameter "cells", and which yields the value defval upon a lookup to a
        key that has not previously been inserted
        '''

        self.table = [None] * cells
        self.size = len(self.table)
        self.num_entries = 0
        self.defval = defval


    def __repr__(self):
        '''Representation of the HashTable object'''

        s = "{"

        for entry in self.table:
            if entry is not None:
                if s != "{":
                    s += ", "
                key, val = (entry.key, entry.val)
                s += "{}: {}".format(key, val)

        s += "}"

        return s


    def values(self):
        '''
        Retrieve each value associated with every key in the HashTable
        '''

        values = [entry.val for entry in self.table if entry is not None]

        return values


    def _hash(self, key):
        '''
        Find the hash index for the 'key' parameter

        Inputs:
            key (str): string to hash

        Returns:
            (int) the hash index for the 'key'
        '''

        first_char = True
        hash_index = None

        for char in key:
            if first_char:
                hash_index = ord(char) % self.size
                first_char = False
            else:
                hash_index = (37 * hash_index + ord(char)) % self.size

        return hash_index


    def lookup(self, key):
        '''
        Retrieve the value associated with the specified key in the hash table,
        or return the default value if it has not previously been inserted.
        '''

        hash_index = self._hash(key)

        if self.table[hash_index] is None:
            return self.defval
        elif self.table[hash_index].key == key:
            return self.table[hash_index].val
        else:
            next_index = hash_index + 1
            while next_index != hash_index:
                if next_index > self.size - 1:
                    next_index = 0
                    continue
                if self.table[next_index] is not None:
                    if self.table[next_index].key == key:
                        return self.table[next_index].val
                next_index += 1

        return self.defval


    def update(self, key, val):
        '''
        Change the value associated with key "key" to value "val".
        If "key" is not currently present in the hash table,  insert it with
        value "val".
        '''

        hash_index = self._hash(key)

        if self.table[hash_index] is None:
            self.table[hash_index] = HashEntry(key, val)
            self.num_entries += 1
        elif self.table[hash_index].key == key:
            self.table[hash_index] = HashEntry(key, val)
        else:
            next_index = hash_index + 1
            while True:
                if next_index > self.size - 1:
                    next_index = 0
                if self.table[next_index] is None:
                    self.table[next_index] = HashEntry(key, val)
                    self.num_entries += 1
                    break
                elif self.table[next_index].key == key:
                    self.table[next_index] = HashEntry(key, val)
                    break
                next_index += 1

        self._check_full()


    def _rehash(self, old_table):
        '''
        Rehash the table when the maximum occupied threshold is exceeded

        Inputs:
            old_table (list): table from the original hash table being rehased

        Modifies the HashTable object
        '''

        for entry in old_table:
            if entry is not None:
                key, val = (entry.key, entry.val)
                self.update(key, val)


    def _check_full(self):
        '''
        Check if the maximum occupied threshold for table is exceeded. If
        threshold is exceeded, increase the size of the table and rehash the
        existing entries. Otherwise, return None
        '''

        if self.num_entries / self.size > TOO_FULL:
            old_table = self.table
            self.table = [None] * (self.size * GROWTH_RATIO)
            self.size = len(self.table)
            self.num_entries = 0
            self._rehash(old_table)
