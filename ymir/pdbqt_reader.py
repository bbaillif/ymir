

class PDBQTReader():
    
    def __init__(self) -> None:
        pass
    
    
    def read(self,
             filepath: str) -> tuple[list[str], list[list[float]]]:
        'ATOM      1  C   UNL     1      -1.017   2.975  24.441  1.00  0.00     0.055 C '
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        symbols = []
        coords = []
        for line in lines:
            line = line.strip()
            if line.startswith('ATOM'):
                elements = line.split()
                assert len(elements) == 12
                symbol = elements[2]
                coord = elements[5:8]
                symbols.append(symbol)
                coords.append(coord)
                
        return symbols, coords
                
    def to_xyz_block(self,
                     filepath: str) -> list[str]:
        xyz_block = []
        symbols, coords = self.read(filepath)
        xyz_block.append(str(len(symbols)))
        xyz_block.append('') # comment line
        for symbol, coord in zip(symbols, coords):
            coord = [str(p) for p in coord]
            line = f'{symbol} {" ".join(coord)}'
            xyz_block.append(line)
        xyz_block = '\n'.join(xyz_block)
        return xyz_block